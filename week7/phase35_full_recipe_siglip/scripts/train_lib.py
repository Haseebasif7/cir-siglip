"""
Phase 35: shared data-loading, batch-building, loss, and evaluation code for
both training stages. Reuses phase 31/32's exact vectorized negative sampler,
category-grouped evaluator structure, and triplet+uniformity loss shape
verbatim wherever the brief doesn't ask for a change -- the only NEW pieces
here are: CPState (piece 1, CP pre-training data), category-token
construction (piece 2), and curriculum negative sampling (piece 3, an
extension of phase 31/32's TrainState.sample_negatives). See
../implementation_notes.md for the full reasoning behind each addition.

Local-only (no Modal): this phase runs entirely on the M4 (MPS if available,
else CPU), per the brief's "run locally first" instruction. See
stage1_training_log.md / stage2_training_log.md for measured wall-clock and
the local-vs-Modal decision actually made.
"""
import json
import os
from pathlib import Path

# uniformity_loss's torch.cdist has no MPS backward kernel yet (as of
# torch 2.6) -- fall back to CPU for that one op rather than avoid MPS
# entirely; set before torch is used anywhere in this process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings/siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data/text_embeddings.npz"
CATEGORY_TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data/category_text_embeddings.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data/training_data.json"
NEGATIVE_CANDIDATES_JSON = PHASE13_DIR / "data/negative_candidates.json"
VAL_BENCHMARK = PHASE23_DIR / "data/cir_val_benchmark.json"
TEST_BENCHMARK = PHASE12_DIR / "data/cir_benchmark.json"  # touched ONLY by 03_final_test_eval.py

NUM_NEGATIVES = 10
QUERY_BATCH = 1024


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_catalog(device):
    """Returns a dict with everything both stages need: base_repr (n_items,
    1536, image+text concat, normalized -- phase 32's winning input_mode,
    held fixed here), id_to_gidx, train/val outfits + category pools, the
    mined hard-negative candidate lists, and the 11 target-category "raw"
    vectors (piece 2) built as concat(zeros(768), category_text_embedding)
    -- the paper's own "x_Img empty || E_text(target description)"
    construction (section 3.2), adapted to this project's SigLIP
    image+text concat convention. See implementation_notes.md."""
    img_npz = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img_npz["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_npz = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    text_ids = [str(a) for a in text_npz["item_ids"]]
    assert text_ids == item_ids, "text_embeddings.npz must be positionally aligned to siglip_base.npz"
    text_emb = text_npz["embeddings"].astype(np.float32)

    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)  # (n_items, 1536)

    cat_text_npz = np.load(CATEGORY_TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    cat_names = [str(c) for c in cat_text_npz["categories"]]
    cat_text_emb = cat_text_npz["embeddings"].astype(np.float32)  # (11, 768), raw SigLIP text features
    zeros_img = np.zeros_like(cat_text_emb)  # paper's "empty image" placeholder half
    cat_raw = np.concatenate([zeros_img, cat_text_emb], axis=1)  # (11, 1536), same shape as an item vector
    cat_raw_t = torch.tensor(cat_raw, device=device)
    cat_row_of = {name: i for i, name in enumerate(cat_names)}

    with open(TRAINING_DATA_JSON) as f:
        td = json.load(f)
    train_outfits = [o for o in td["train_outfits"] if len(o["items"]) >= 2]
    val_outfits = [o for o in td["val_outfits"] if len(o["items"]) >= 2]
    train_items_by_cat = td["train_items_by_category"]
    val_items_by_cat = td["val_items_by_category"]

    with open(NEGATIVE_CANDIDATES_JSON) as f:
        neg_cand = json.load(f)  # {"train": {item_id: [cand_id, ...20...]}, "val": {...}}

    return {
        "item_ids": item_ids, "id_to_gidx": id_to_gidx, "base_repr": base_repr,
        "cat_raw_t": cat_raw_t, "cat_row_of": cat_row_of, "cat_names": cat_names,
        "train_outfits": train_outfits, "val_outfits": val_outfits,
        "train_items_by_cat": train_items_by_cat, "val_items_by_cat": val_items_by_cat,
        "neg_cand_train": neg_cand["train"], "neg_cand_val": neg_cand["val"],
    }


def category_tokens_all(model, cat_raw_t):
    """(11, d_model) normalized category tokens -- recomputed every time
    it's called since self.proj is trainable in stage 2; cheap (11 rows)."""
    return model.encode_item_tokens(cat_raw_t)


# --------------------------------------------------------------------------
# Stage 2: retrieval fine-tuning with curriculum negative sampling (piece 3)
# --------------------------------------------------------------------------

class RetrievalTrainState:
    """Vectorized category-restricted negative sampler, extended from phase
    31/32's TrainState with CURRICULUM mixing: each sample's num_negatives
    slots are split between the existing random-same-category pool (the
    "easy" stage of the paper's two-stage curriculum, section 3.2.2 --
    "sample the negatives from the same high-level category as the
    positive") and a static mined-hard same-category pool (the "harder"
    stage -- "sample harder negatives from more fine-grained categories"),
    per a hard_fraction that the caller advances across epochs. See
    implementation_notes.md for why this project's single-granularity
    11-category system maps the paper's two-level category distinction onto
    this project's existing mined-vs-random negative distinction instead
    (an honest, documented adaptation, not a literal high/fine-grained
    split)."""

    def __init__(self, items_by_cat, id_to_gidx, neg_cand, seed):
        self.rng_np = np.random.default_rng(seed)
        self.item_cat = {}
        self.cat_pool_gidx = {}
        for cat, items in items_by_cat.items():
            for i in items:
                self.item_cat[i] = cat
            self.cat_pool_gidx[cat] = np.array(
                [id_to_gidx[i] for i in items if i in id_to_gidx], dtype=np.int64
            )
        self.neg_cand = neg_cand
        self.id_to_gidx = id_to_gidx

    def _sample_random(self, category, exclude_gidx_set, k):
        pool = self.cat_pool_gidx.get(category)
        if pool is None or len(pool) == 0 or k <= 0:
            return []
        n_pool = len(pool)
        chosen, seen = [], set()
        attempts = 0
        draw_size = min(n_pool, k + 8)
        while len(chosen) < k and attempts < 20:
            cand_pos = self.rng_np.integers(0, n_pool, size=draw_size)
            for g in pool[cand_pos]:
                gi = int(g)
                if gi not in exclude_gidx_set and gi not in seen:
                    seen.add(gi)
                    chosen.append(gi)
                    if len(chosen) >= k:
                        break
            attempts += 1
        return chosen

    def _sample_hard(self, target_id, exclude_gidx_set, k):
        if k <= 0:
            return []
        cands = self.neg_cand.get(target_id)
        if not cands:
            return []
        cand_gidx = [self.id_to_gidx[c] for c in cands if c in self.id_to_gidx]
        cand_gidx = [g for g in cand_gidx if g not in exclude_gidx_set]
        if not cand_gidx:
            return []
        if len(cand_gidx) <= k:
            return cand_gidx
        idx = self.rng_np.choice(len(cand_gidx), size=k, replace=False)
        return [cand_gidx[i] for i in idx]

    def sample_negatives(self, target_id, category, exclude_gidx_set, num_negatives, hard_fraction):
        n_hard = int(round(num_negatives * hard_fraction))
        hard = self._sample_hard(target_id, exclude_gidx_set, n_hard)
        n_random = num_negatives - len(hard)
        excl2 = exclude_gidx_set | set(hard)
        rand = self._sample_random(category, excl2, n_random)
        # if the hard pool came up short (missing/exhausted candidates),
        # top back up with random negatives so the curriculum's INTENDED
        # hard_fraction is a target, not a hard guarantee -- documented in
        # implementation_notes.md.
        if len(hard) + len(rand) < num_negatives:
            rand += self._sample_random(category, excl2 | set(rand), num_negatives - len(hard) - len(rand))
        return np.array(hard + rand, dtype=np.int64)

    def make_sample(self, outfit_record, num_negatives, hard_fraction):
        items = outfit_record["items"]
        target = items[int(self.rng_np.integers(0, len(items)))]
        context = [i for i in items if i != target]
        target_cat = self.item_cat[target]
        exclude_gidx = {self.id_to_gidx[i] for i in items if i in self.id_to_gidx}
        neg_gidx = self.sample_negatives(target, target_cat, exclude_gidx, num_negatives, hard_fraction)
        return {
            "context_gidx": np.array([self.id_to_gidx[i] for i in context if i in self.id_to_gidx], dtype=np.int64),
            "target_gidx": self.id_to_gidx[target],
            "neg_gidx": neg_gidx,
            "target_cat": target_cat,
        }


def build_retrieval_batch(samples, base_repr, cat_row_of, device):
    B = len(samples)
    lengths = [len(s["context_gidx"]) for s in samples]
    Lmax = max(max(lengths), 1)
    ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
    ctx_mask = np.ones((B, Lmax), dtype=bool)
    for i, s in enumerate(samples):
        L = len(s["context_gidx"])
        ctx_gidx[i, :L] = s["context_gidx"]
        ctx_mask[i, :L] = False

    target_gidx = np.array([s["target_gidx"] for s in samples], dtype=np.int64)
    cat_idx = np.array([cat_row_of[s["target_cat"]] for s in samples], dtype=np.int64)

    Mmax = max((len(s["neg_gidx"]) for s in samples), default=1)
    Mmax = max(Mmax, 1)
    neg_gidx = np.zeros((B, Mmax), dtype=np.int64)
    neg_mask = np.zeros((B, Mmax), dtype=bool)
    for i, s in enumerate(samples):
        M = len(s["neg_gidx"])
        if M > 0:
            neg_gidx[i, :M] = s["neg_gidx"]
            neg_mask[i, :M] = True

    return (
        base_repr[torch.tensor(ctx_gidx, device=device)],
        torch.tensor(ctx_mask, device=device),
        base_repr[torch.tensor(target_gidx, device=device)],
        base_repr[torch.tensor(neg_gidx, device=device)],
        torch.tensor(neg_mask, device=device),
        torch.tensor(cat_idx, device=device),
    )


def category_negative_triplet_loss(query_emb, answer_emb, neg_emb, neg_mask, margin):
    pos = torch.norm(query_emb - answer_emb, p=2, dim=-1)
    d_negs = torch.norm(query_emb.unsqueeze(1) - neg_emb, p=2, dim=-1)
    d_negs = d_negs.masked_fill(~neg_mask, float("inf"))
    hardest_neg, _ = d_negs.min(dim=1)
    loss = F.relu(pos - hardest_neg + margin)
    return loss.mean(), pos.mean().item(), hardest_neg.mean().item()


def compute_retrieval_batch_loss(model, cat_raw_t, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask,
                                  cat_idx, margin, uniformity_weight, uniformity_loss_fn):
    B, Lmax, D = ctx_vecs.shape
    ctx_tokens = model.encode_item_tokens(ctx_vecs.reshape(B * Lmax, D)).reshape(B, Lmax, -1)
    cat_tok_all = category_tokens_all(model, cat_raw_t)          # (11, d_model)
    cat_tok = cat_tok_all[cat_idx]                                 # (B, d_model), piece 2
    query_emb = model.embed_query_targeted(ctx_tokens, ctx_mask, cat_tok)

    target_tokens = model.encode_item_tokens(target_vecs)
    answer_emb = model.embed_item_alone(target_tokens)

    Bn, M, Dn = neg_vecs.shape
    neg_tokens = model.encode_item_tokens(neg_vecs.reshape(Bn * M, Dn)).reshape(Bn, M, -1)
    neg_emb = model.embed_item_alone(neg_tokens.reshape(Bn * M, -1)).reshape(Bn, M, -1)

    triplet_loss, d_pos, d_neg = category_negative_triplet_loss(query_emb, answer_emb, neg_emb, neg_mask, margin)
    total_loss = triplet_loss
    if uniformity_weight > 0:
        total_loss = total_loss + uniformity_weight * uniformity_loss_fn(answer_emb)
    return total_loss, d_pos, d_neg


_bench_cache = {}


def load_bench(bench_key):
    if bench_key not in _bench_cache:
        path = VAL_BENCHMARK if bench_key == "val" else None
        assert path is not None, "only 'val' may be loaded here -- test benchmark is loaded exclusively by 03_final_test_eval.py"
        with open(path) as f:
            b = json.load(f)
        _bench_cache[bench_key] = (b["pools"], b["queries"])
    return _bench_cache[bench_key]


def evaluate_recall_targeted(model, base_repr, id_to_gidx, cat_raw_t, cat_row_of, device,
                              bench_key="val", ks=(10, 30, 50), bench_override=None):
    """Category-conditioned evaluator (piece 2): identical structure to
    phase 31/32's evaluate_recall_gpu (candidates precomputed catalog-wide
    via embed_item_alone, queries batched per category), but the query
    embedding now uses embed_query_targeted with the query's OWN target
    category's token (single vector, broadcast across that category's whole
    query batch -- embed_set already supports 1-d broadcast)."""
    if bench_override is not None:
        pools, queries = bench_override
    else:
        pools, queries = load_bench(bench_key)
    model.eval()
    with torch.no_grad():
        item_tokens_all = model.encode_item_tokens(base_repr)
        cand_emb_all = model.embed_item_alone(item_tokens_all).cpu().numpy()
        cat_tok_all = category_tokens_all(model, cat_raw_t)  # (11, d_model)

        from collections import defaultdict
        by_cat = defaultdict(list)
        for qi, q in enumerate(queries):
            by_cat[q["category"]].append(qi)

        hits = {k: 0 for k in ks}
        n_total, n_skipped = 0, 0
        for cat, qidxs in by_cat.items():
            if cat not in pools or cat not in cat_row_of:
                n_skipped += len(qidxs)
                continue
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_idx = [id_to_gidx[i] for i in pool_ids if i in id_to_gidx]
            pool_emb = cand_emb_all[pool_idx]
            cat_tok = cat_tok_all[cat_row_of[cat]]  # (d_model,) broadcast

            kept_qidx, ctx_lists, tpos = [], [], []
            for qi in qidxs:
                q = queries[qi]
                ctx = [id_to_gidx[i] for i in q["query_items"] if i in id_to_gidx]
                if not ctx or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                kept_qidx.append(qi)
                ctx_lists.append(ctx)
                tpos.append(pool_pos[q["target_item"]])
            if not kept_qidx:
                continue

            q_embs = []
            for start in range(0, len(ctx_lists), QUERY_BATCH):
                chunk = ctx_lists[start:start + QUERY_BATCH]
                Lmax = max(len(c) for c in chunk)
                B = len(chunk)
                ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
                mask = np.ones((B, Lmax), dtype=bool)
                for i, c in enumerate(chunk):
                    ctx_gidx[i, :len(c)] = c
                    mask[i, :len(c)] = False
                ctx_t = base_repr[torch.tensor(ctx_gidx, device=device)]
                mask_t = torch.tensor(mask, device=device)
                Bc, Lc, Dc = ctx_t.shape
                tokens = model.encode_item_tokens(ctx_t.reshape(Bc * Lc, Dc)).reshape(Bc, Lc, -1)
                q_emb = model.embed_query_targeted(tokens, mask_t, cat_tok)
                q_embs.append(q_emb.cpu().numpy())
            q_mat = np.concatenate(q_embs, axis=0)

            sims = q_mat @ pool_emb.T
            tpos_arr = np.array(tpos)
            tsims = sims[np.arange(len(kept_qidx)), tpos_arr]
            ranks = (sims >= tsims[:, None]).sum(axis=1)
            n_total += len(kept_qidx)
            for k in ks:
                hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


# --------------------------------------------------------------------------
# Stage 1: compatibility-prediction (CP) pre-training (piece 1)
# --------------------------------------------------------------------------

class CPState:
    """Builds real (label 1) and shuffled-fake (label 0) whole-item-set
    samples for CP pre-training -- "negative outfits (items shuffled across
    different real outfits to create fake combinations)" per the brief
    (the paper itself doesn't specify negative construction for this task,
    see implementation_notes.md). A fake outfit of size k draws k items,
    each from an independently, uniformly chosen DIFFERENT real outfit --
    drawn fresh every epoch (same "fresh negatives every call" convention
    as phase 34's random-negative sampler)."""

    def __init__(self, outfits, seed):
        self.outfits = outfits
        self.n = len(outfits)
        self.rng_np = np.random.default_rng(seed)

    def make_fake_items(self, size):
        items = []
        for _ in range(size):
            j = int(self.rng_np.integers(0, self.n))
            o = self.outfits[j]
            items.append(o["items"][int(self.rng_np.integers(0, len(o["items"])))])
        return items

    def make_batch_lists(self, outfit_batch):
        """Returns (item_id_lists, labels) -- one real + one fake per outfit
        in outfit_batch, so a batch of B outfits yields 2B sequences."""
        item_lists, labels = [], []
        for o in outfit_batch:
            item_lists.append(o["items"])
            labels.append(1.0)
            item_lists.append(self.make_fake_items(len(o["items"])))
            labels.append(0.0)
        return item_lists, labels


def build_cp_batch(item_id_lists, labels, base_repr, id_to_gidx, device):
    B = len(item_id_lists)
    lengths = [len(lst) for lst in item_id_lists]
    Lmax = max(lengths)
    gidx = np.zeros((B, Lmax), dtype=np.int64)
    mask = np.ones((B, Lmax), dtype=bool)
    for i, lst in enumerate(item_id_lists):
        idxs = [id_to_gidx[x] for x in lst if x in id_to_gidx]
        L = len(idxs)
        gidx[i, :L] = idxs
        mask[i, :L] = False
    item_vecs = base_repr[torch.tensor(gidx, device=device)]
    pad_mask = torch.tensor(mask, device=device)
    label_t = torch.tensor(labels, dtype=torch.float32, device=device)
    return item_vecs, pad_mask, label_t


def compute_cp_loss(model, cp_focal_loss_fn, item_vecs, pad_mask, label_t):
    B, Lmax, D = item_vecs.shape
    item_tokens = model.encode_item_tokens(item_vecs.reshape(B * Lmax, D)).reshape(B, Lmax, -1)
    logits = model.cp_score(item_tokens, pad_mask)
    loss = cp_focal_loss_fn(logits, label_t)
    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == label_t).float().mean().item()
    return loss, acc, logits

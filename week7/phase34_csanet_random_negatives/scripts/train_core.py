"""
Phase 34: phase 33's vectorized training/eval core for CSA-Net-on-SigLIP,
with EXACTLY ONE change from phase 33's copy -- negative sampling.

WHAT CHANGED, PRECISELY: TrainState._sample_negatives no longer reads
`week4/phase13_csa_net_baseline/data/negative_candidates.json` (phase 13's
original mined, SigLIP-nearest-neighbor hard negatives). It instead draws
`num_negatives` uniform-random items from the positive's own same-category
pool (`items_by_cat[category]`), excluding the outfit's own items -- exactly
the category-restricted, uniform-random scheme phase 14b's run 2 adopted for
OutfitTransformer (0.0051 mined -> 0.0588 random, phase 14b's own
`training_log.md`) and this project's own model has used since phases 7-9.
Negatives are still redrawn fresh every epoch (make_sample is called once per
sample per epoch, same as phase 33 -- this was already true there, not new
here). load_catalog() no longer takes a negative_candidates_path argument at
all, since nothing in this phase's TrainState reads that file anymore.

WHAT DID NOT CHANGE, for the record (see negative_sampling_change.md for the
full diff against phase 33's file): model.py (byte-identical copy), the loss
function shape (triplet margin, min-aggregation over negatives, uniformity
regularizer -- same MARGIN=0.3, UNIFORMITY_WEIGHT=1.0 phase 33 actually
trained with, matching phase 33's own modal_app.py DEFAULT_CONFIG/
04_train_seed.py WINNING_CONFIG, not the phase 34 brief's own approximate
"uniformity weight at 0.1" aside -- that value was phase 31's OutfitTransformer
tuning winner, not anything CSA-Net was ever tuned to; preserved here exactly
as phase 33 measured it), text integration (image+text cascaded concat,
identical construction), checkpoint selection (validation Recall@10, patience
5, min_delta 0.0005), compute_batch_loss, evaluate_recall, and
evaluate_recall_ensemble (all byte-identical to phase 33's file, since the
CATEGORY-SHARING ASSUMPTION documented below still holds under random
same-category negatives -- they share the positive's category by
construction, same as mined negatives did).

CATEGORY-SHARING ASSUMPTION (still holds, unchanged from phase 33): negatives
are drawn from the positive's own same-category pool, so they share the
positive's category by construction -- `cat_t` (target/positive category) is
one shared value per sample, used for both the positive AND every negative's
conditioning. This is what makes the vectorization tractable (one cat_t per
sample, broadcast, rather than a per-negative category), and was true of
phase 33's mined negatives for the same reason (mined candidates were
themselves restricted to same-category items).
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CSANetSigLIP, pairwise_distance, uniformity_loss

NUM_NEGATIVES = 10
UNIFORMITY_WEIGHT = 1.0
MARGIN = 0.3


class TrainState:
    """Category pools are real multisets (an item appears once per outfit
    containing it) -- category assignment (item_cat) is identical to phase
    33/13b's own TrainState. The ONLY change from phase 33 is
    _sample_negatives below: random same-category draw instead of a mined-
    candidate lookup. Seeded via numpy/random for consistency with phase
    31/32/33's own seed-plumbing fix."""

    def __init__(self, id_to_gidx, item_cat, cat_to_idx, train_items_by_cat, val_items_by_cat,
                 split_key, seed):
        self.rng = random.Random(seed)
        self.id_to_gidx = id_to_gidx
        self.item_cat = item_cat
        self.cat_to_idx = cat_to_idx
        self.items_by_cat = train_items_by_cat if split_key == "train" else val_items_by_cat
        self.split_key = split_key

    def _sample_negatives(self, positive_id, category, exclude, num_negatives):
        """Phase 34's single changed line: category-restricted, uniform-
        random negatives, drawn fresh every call (i.e. fresh every epoch,
        since make_sample is called once per sample per epoch) -- exactly
        phase 14b's run 2 scheme for OutfitTransformer, no mined candidates
        involved at all."""
        pool = [i for i in self.items_by_cat[category] if i not in exclude]
        if not pool:
            return []
        return self.rng.sample(pool, min(num_negatives, len(pool)))

    def make_sample(self, outfit_record, num_negatives=NUM_NEGATIVES):
        items = outfit_record["items"]
        positive = self.rng.choice(items)
        context = [i for i in items if i != positive]
        pos_cat = self.item_cat[positive]
        exclude = set(items)
        negatives = self._sample_negatives(positive, pos_cat, exclude, num_negatives)
        return {
            "context_gidx": [self.id_to_gidx[i] for i in context if i in self.id_to_gidx],
            "context_cat_idx": [self.cat_to_idx[self.item_cat[i]] for i in context if i in self.id_to_gidx],
            "positive_gidx": self.id_to_gidx[positive],
            "positive_cat_idx": self.cat_to_idx[pos_cat],
            "neg_gidx": [self.id_to_gidx[i] for i in negatives if i in self.id_to_gidx],
        }


def build_batch(samples, device):
    """Pads a batch of make_sample() dicts into tensors. Returns everything
    compute_batch_loss needs -- no python-level per-sample work happens
    downstream of this function. Byte-identical to phase 33."""
    B = len(samples)
    lengths = [len(s["context_gidx"]) for s in samples]
    Lmax = max(max(lengths), 1)
    Mmax = max(max(len(s["neg_gidx"]) for s in samples), 1)

    ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
    ctx_cat_idx = np.zeros((B, Lmax), dtype=np.int64)
    ctx_mask = np.zeros((B, Lmax), dtype=bool)  # True = valid
    for i, s in enumerate(samples):
        L = len(s["context_gidx"])
        ctx_gidx[i, :L] = s["context_gidx"]
        ctx_cat_idx[i, :L] = s["context_cat_idx"]
        ctx_mask[i, :L] = True

    pos_gidx = np.array([s["positive_gidx"] for s in samples], dtype=np.int64)
    pos_cat_idx = np.array([s["positive_cat_idx"] for s in samples], dtype=np.int64)

    neg_gidx = np.zeros((B, Mmax), dtype=np.int64)
    neg_mask = np.zeros((B, Mmax), dtype=bool)
    for i, s in enumerate(samples):
        M = len(s["neg_gidx"])
        if M > 0:
            neg_gidx[i, :M] = s["neg_gidx"]
            neg_mask[i, :M] = True

    return {
        "ctx_gidx": torch.tensor(ctx_gidx, device=device),
        "ctx_cat_idx": torch.tensor(ctx_cat_idx, device=device),
        "ctx_mask": torch.tensor(ctx_mask, device=device),
        "pos_gidx": torch.tensor(pos_gidx, device=device),
        "pos_cat_idx": torch.tensor(pos_cat_idx, device=device),
        "neg_gidx": torch.tensor(neg_gidx, device=device),
        "neg_mask": torch.tensor(neg_mask, device=device),
    }


def compute_batch_loss(model, base_repr, batch, num_categories, margin, uniformity_weight, aggregation="min"):
    """Vectorized triplet+uniformity loss. Byte-identical to phase 33 --
    the negative-sampling change happens entirely upstream, in
    TrainState._sample_negatives; nothing here depends on where negatives
    came from, only on the (already-built) batch tensors."""
    ctx_gidx, ctx_cat_idx, ctx_mask = batch["ctx_gidx"], batch["ctx_cat_idx"], batch["ctx_mask"]
    pos_gidx, pos_cat_idx = batch["pos_gidx"], batch["pos_cat_idx"]
    neg_gidx, neg_mask = batch["neg_gidx"], batch["neg_mask"]
    B, Lmax = ctx_gidx.shape
    Mmax = neg_gidx.shape[1]
    C = num_categories
    device = ctx_gidx.device

    x_ctx = model.encode_feature(base_repr[ctx_gidx])          # (B, Lmax, D)
    x_pos = model.encode_feature(base_repr[pos_gidx])          # (B, D)
    x_neg = model.encode_feature(base_repr[neg_gidx])          # (B, Mmax, D)

    cat_s_flat = F.one_hot(ctx_cat_idx.reshape(-1), C).float()                          # (B*Lmax, C)
    cat_t_flat = F.one_hot(pos_cat_idx, C).float().unsqueeze(1).expand(B, Lmax, C).reshape(-1, C)  # (B*Lmax, C)

    f_ctx = model.embed_from_feature(x_ctx.reshape(B * Lmax, -1), cat_s_flat, cat_t_flat).reshape(B, Lmax, -1)

    x_pos_bcast = x_pos.unsqueeze(1).expand(B, Lmax, -1).reshape(B * Lmax, -1)
    f_pos = model.embed_from_feature(x_pos_bcast, cat_s_flat, cat_t_flat).reshape(B, Lmax, -1)

    # negatives: broadcast across BOTH Lmax (context positions) and Mmax (negative slots)
    x_neg_bcast = x_neg.unsqueeze(1).expand(B, Lmax, Mmax, -1).reshape(B * Lmax * Mmax, -1)
    cat_s_neg = cat_s_flat.unsqueeze(1).expand(B * Lmax, Mmax, C).reshape(-1, C)
    cat_t_neg = cat_t_flat.unsqueeze(1).expand(B * Lmax, Mmax, C).reshape(-1, C)
    f_neg = model.embed_from_feature(x_neg_bcast, cat_s_neg, cat_t_neg).reshape(B, Lmax, Mmax, -1)

    d_pos = pairwise_distance(f_ctx, f_pos)                    # (B, Lmax)
    d_neg = ((f_ctx.unsqueeze(2) - f_neg) ** 2).sum(dim=-1)    # (B, Lmax, Mmax)

    n_ctx = ctx_mask.sum(dim=1).clamp(min=1)                                   # (B,)
    D_pos = (d_pos * ctx_mask).sum(dim=1) / n_ctx                              # (B,)
    D_neg_per_neg = (d_neg * ctx_mask.unsqueeze(-1)).sum(dim=1) / n_ctx.unsqueeze(-1)  # (B, Mmax)
    D_neg_per_neg = D_neg_per_neg.masked_fill(~neg_mask, float("inf"))

    if aggregation == "min":
        D_neg_agg = D_neg_per_neg.min(dim=-1).values
    else:
        D_neg_agg = D_neg_per_neg.sum(dim=-1) / neg_mask.sum(dim=-1).clamp(min=1)

    ranking_loss = F.relu(D_pos - D_neg_agg + margin).mean()

    representative = (f_pos * ctx_mask.unsqueeze(-1)).sum(dim=1) / n_ctx.unsqueeze(-1)  # (B, D)
    representative = F.normalize(representative, p=2, dim=-1)
    uniformity = uniformity_loss(representative) if uniformity_weight > 0 else torch.tensor(0.0, device=device)

    total_loss = ranking_loss + uniformity_weight * uniformity
    return total_loss, D_pos.mean().item(), D_neg_agg.mean().item()


def evaluate_recall(model, base_repr, id_to_gidx, cat_to_idx, item_cat_lookup, pools, queries,
                     device, ks=(10, 30, 50), query_chunk=512):
    """Vectorized CIR evaluator. Byte-identical to phase 33 -- evaluation
    has nothing to do with training-time negative sampling."""
    from collections import defaultdict
    C = len(cat_to_idx)
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    model.eval()
    with torch.no_grad():
        for cat, qidxs in by_cat.items():
            if cat not in pools:
                n_skipped += len(qidxs)
                continue
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_gidx = [id_to_gidx[i] for i in pool_ids if i in id_to_gidx]
            if len(pool_gidx) != len(pool_ids):
                n_skipped += len(qidxs)
                continue
            x_pool = model.encode_feature(base_repr[torch.tensor(pool_gidx, device=device)])  # (P, D)
            cat_t_fixed = torch.zeros(len(pool_gidx), C, device=device)
            cat_t_fixed[:, cat_to_idx[cat]] = 1.0
            cand_all = model.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed)  # (P, C, D)

            kept_qidx, ctx_lists, ctx_cat_lists, tpos = [], [], [], []
            for qi in qidxs:
                q = queries[qi]
                items = [i for i in q["query_items"] if i in id_to_gidx]
                if not items or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                kept_qidx.append(qi)
                ctx_lists.append([id_to_gidx[i] for i in items])
                ctx_cat_lists.append([cat_to_idx[item_cat_lookup[i]] for i in items])
                tpos.append(pool_pos[q["target_item"]])
            if not kept_qidx:
                continue

            for start in range(0, len(kept_qidx), query_chunk):
                sl = slice(start, start + query_chunk)
                chunk_ctx, chunk_cat, chunk_tpos = ctx_lists[sl], ctx_cat_lists[sl], tpos[sl]
                B = len(chunk_ctx)
                Lmax = max(len(c) for c in chunk_ctx)
                ctx_gidx_np = np.zeros((B, Lmax), dtype=np.int64)
                ctx_cat_np = np.zeros((B, Lmax), dtype=np.int64)
                ctx_mask_np = np.zeros((B, Lmax), dtype=bool)
                for i, (c, cc) in enumerate(zip(chunk_ctx, chunk_cat)):
                    L = len(c)
                    ctx_gidx_np[i, :L] = c
                    ctx_cat_np[i, :L] = cc
                    ctx_mask_np[i, :L] = True
                ctx_gidx_t = torch.tensor(ctx_gidx_np, device=device)
                ctx_cat_t = torch.tensor(ctx_cat_np, device=device)
                ctx_mask_t = torch.tensor(ctx_mask_np, device=device)

                x_ctx = model.encode_feature(base_repr[ctx_gidx_t])  # (B, Lmax, D)
                cat_s_flat = F.one_hot(ctx_cat_t.reshape(-1), C).float()
                cat_t_flat = torch.zeros(B * Lmax, C, device=device)
                cat_t_flat[:, cat_to_idx[cat]] = 1.0
                f_ctx = model.embed_from_feature(x_ctx.reshape(B * Lmax, -1), cat_s_flat, cat_t_flat).reshape(B, Lmax, -1)

                P = cand_all.shape[0]
                dist_sum = torch.zeros(B, P, device=device)
                for c in range(C):
                    sel = (ctx_cat_t == c) & ctx_mask_t  # (B, Lmax)
                    if not sel.any():
                        continue
                    sims_c = torch.einsum("bld,pd->blp", f_ctx, cand_all[:, c, :])  # (B, Lmax, P)
                    dist_c = (2.0 - 2.0 * sims_c) * sel.unsqueeze(-1).float()
                    dist_sum += dist_c.sum(dim=1)

                n_ctx = ctx_mask_t.sum(dim=1, keepdim=True).clamp(min=1).float()
                dist_avg = (dist_sum / n_ctx).cpu().numpy()
                tpos_arr = np.array(chunk_tpos)
                target_dist = dist_avg[np.arange(B), tpos_arr]
                ranks = (dist_avg <= target_dist[:, None]).sum(axis=1)

                n_total += B
                for k in ks:
                    hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


def load_catalog(embeddings_npz, training_data_path, text_embeddings_npz=None, metadata_path=None, device="cpu"):
    """Loads everything shared across train/val TrainStates: base_repr
    (image-only 768-d, or image+text 1536-d if text_embeddings_npz given --
    identical construction to phase 27/28/31/32/33) and category tables.
    NO negative_candidates_path argument -- phase 34's whole point is that
    nothing here reads the mined-candidate file at all anymore."""
    img = np.load(embeddings_npz, allow_pickle=True)
    item_ids = [str(a) for a in img["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    image_t = torch.tensor(image_emb, device=device)

    if text_embeddings_npz is not None:
        txt = np.load(text_embeddings_npz, allow_pickle=True)
        assert [str(a) for a in txt["item_ids"]] == item_ids, "text embeddings must align to image embeddings"
        text_t = torch.tensor(txt["embeddings"].astype(np.float32), device=device)
        base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)
        in_dim = base_repr.shape[1]
    else:
        base_repr = image_t
        in_dim = base_repr.shape[1]

    with open(training_data_path) as f:
        td = json.load(f)
    categories = td["categories"]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    train_outfits = [o for o in td["train_outfits"] if len(o["items"]) >= 2]
    val_outfits = [o for o in td["val_outfits"] if len(o["items"]) >= 2]
    train_items_by_cat = td["train_items_by_category"]
    val_items_by_cat = td["val_items_by_category"]
    item_cat = {}
    for cat, items in train_items_by_cat.items():
        for i in items:
            item_cat[i] = cat
    for cat, items in val_items_by_cat.items():
        for i in items:
            item_cat[i] = cat

    # Fallback for items outside the train/val outfit universe (e.g. some
    # test-benchmark query-context items) -- exactly phase 13b/33's own
    # precedent: look up semantic_category from the raw Polyvore item
    # metadata for anything item_cat doesn't already cover. Only needed for
    # the final test-benchmark eval -- see 04_final_test_eval.py.
    if metadata_path is not None:
        with open(metadata_path) as f:
            meta = json.load(f)
        for item_id, v in meta.items():
            cat = v.get("semantic_category")
            if item_id not in item_cat and cat in categories:
                item_cat[item_id] = cat

    return {
        "base_repr": base_repr, "in_dim": in_dim, "id_to_gidx": id_to_gidx,
        "categories": categories, "cat_to_idx": cat_to_idx, "item_cat": item_cat,
        "train_outfits": train_outfits, "val_outfits": val_outfits,
        "train_items_by_cat": train_items_by_cat, "val_items_by_cat": val_items_by_cat,
    }


def run_training(catalog, device, max_epochs, batch_size, lr, patience, margin=MARGIN,
                  uniformity_weight=UNIFORMITY_WEIGHT, num_negatives=NUM_NEGATIVES,
                  seed=42, min_delta=0.0005, selection_metric="recall10", eval_every=1,
                  val_benchmark=None, log_every=100):
    """selection_metric: "recall10" (this phase's only mode -- phase 33's
    val_loss fidelity-check mode isn't needed again here, step 1 already
    settled that question for CSA-Net). val_benchmark: (pools, queries)
    tuple, required for recall-based selection."""
    torch.manual_seed(seed)
    base_repr = catalog["base_repr"]
    id_to_gidx = catalog["id_to_gidx"]
    cat_to_idx = catalog["cat_to_idx"]
    item_cat = catalog["item_cat"]
    num_categories = len(cat_to_idx)

    train_state = TrainState(id_to_gidx, item_cat, cat_to_idx, catalog["train_items_by_cat"],
                              catalog["val_items_by_cat"], "train", seed=seed)
    val_state = TrainState(id_to_gidx, item_cat, cat_to_idx, catalog["train_items_by_cat"],
                            catalog["val_items_by_cat"], "val", seed=42)
    val_samples_fixed = [val_state.make_sample(o, num_negatives) for o in catalog["val_outfits"]]

    model = CSANetSigLIP(num_categories=num_categories, siglip_dim=catalog["in_dim"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(catalog["train_outfits"]) // batch_size)
    total_steps = max_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    curve = []
    best_selection_score = -1.0
    best_recall10 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    step = 0
    t_start = __import__("time").time()

    train_outfits = catalog["train_outfits"]
    for epoch in range(max_epochs):
        model.train()
        order = list(range(len(train_outfits)))
        train_state.rng.shuffle(order)
        epoch_losses, epoch_dpos, epoch_dneg = [], [], []
        for b_start in range(0, len(order), batch_size):
            b_idx = order[b_start:b_start + batch_size]
            if len(b_idx) < 2:
                continue
            samples = [train_state.make_sample(train_outfits[i], num_negatives) for i in b_idx]
            batch = build_batch(samples, device)
            optimizer.zero_grad()
            loss, d_pos, d_neg = compute_batch_loss(model, base_repr, batch, num_categories, margin, uniformity_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            epoch_dpos.append(d_pos)
            epoch_dneg.append(d_neg)
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f} "
                      f"D_pos={d_pos:.4f} D_neg={d_neg:.4f}", flush=True)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        model.eval()
        val_losses, val_dpos, val_dneg = [], [], []
        with torch.no_grad():
            for b_start in range(0, len(val_samples_fixed), batch_size):
                vb = val_samples_fixed[b_start:b_start + batch_size]
                if len(vb) < 2:
                    continue
                batch = build_batch(vb, device)
                loss, d_pos, d_neg = compute_batch_loss(model, base_repr, batch, num_categories, margin, uniformity_weight)
                val_losses.append(loss.item())
                val_dpos.append(d_pos)
                val_dneg.append(d_neg)
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        do_eval = (epoch % eval_every == 0) or (epoch == max_epochs - 1)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_D_pos": float(np.mean(val_dpos)) if val_dpos else 0.0,
               "val_D_neg": float(np.mean(val_dneg)) if val_dneg else 0.0}
        if do_eval and val_benchmark is not None:
            pools, queries = val_benchmark
            recall, n_total, n_skipped = evaluate_recall(model, base_repr, id_to_gidx, cat_to_idx, item_cat,
                                                            pools, queries, device)
            row.update({"recall10": recall[10], "recall30": recall[30], "recall50": recall[50]})

            improved = recall[10] > best_selection_score + min_delta
            if improved:
                best_selection_score = recall[10]
                best_recall10 = recall[10]
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    curve.append(row)
                    break
        curve.append(row)

    wall_time = __import__("time").time() - t_start
    return {
        "curve": curve, "best_epoch": best_epoch, "best_recall10": best_recall10,
        "n_epochs_run": len(curve), "wall_time_sec": wall_time, "n_params": n_params,
        "best_state": best_state, "model_kwargs": {"num_categories": num_categories, "siglip_dim": catalog["in_dim"]},
    }


def evaluate_recall_ensemble(models, base_repr, id_to_gidx, cat_to_idx, item_cat_lookup, pools, queries,
                              device, ks=(10, 30, 50), query_chunk=512):
    """Score-averaging ensemble evaluator: averages per-member DISTANCE
    scores (CSA-Net ranks candidates by ascending squared distance, not
    cosine similarity like OutfitTransformer -- same score-averaging
    PRINCIPLE as phases 26/28/30/32/33, just applied to the score this
    architecture's own ranking is based on. Never averages embeddings --
    independently-trained models share no coordinate system). Byte-identical
    to phase 33."""
    from collections import defaultdict
    C = len(cat_to_idx)
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    for m in models:
        m.eval()

    with torch.no_grad():
        for cat, qidxs in by_cat.items():
            if cat not in pools:
                n_skipped += len(qidxs)
                continue
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_gidx = [id_to_gidx[i] for i in pool_ids if i in id_to_gidx]
            if len(pool_gidx) != len(pool_ids):
                n_skipped += len(qidxs)
                continue

            cand_all_per_member = []
            for m in models:
                x_pool = m.encode_feature(base_repr[torch.tensor(pool_gidx, device=device)])
                cat_t_fixed = torch.zeros(len(pool_gidx), C, device=device)
                cat_t_fixed[:, cat_to_idx[cat]] = 1.0
                cand_all_per_member.append(m.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed))

            kept_qidx, ctx_lists, ctx_cat_lists, tpos = [], [], [], []
            for qi in qidxs:
                q = queries[qi]
                items = [i for i in q["query_items"] if i in id_to_gidx]
                if not items or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                kept_qidx.append(qi)
                ctx_lists.append([id_to_gidx[i] for i in items])
                ctx_cat_lists.append([cat_to_idx[item_cat_lookup[i]] for i in items])
                tpos.append(pool_pos[q["target_item"]])
            if not kept_qidx:
                continue

            for start in range(0, len(kept_qidx), query_chunk):
                sl = slice(start, start + query_chunk)
                chunk_ctx, chunk_cat, chunk_tpos = ctx_lists[sl], ctx_cat_lists[sl], tpos[sl]
                B = len(chunk_ctx)
                Lmax = max(len(c) for c in chunk_ctx)
                ctx_gidx_np = np.zeros((B, Lmax), dtype=np.int64)
                ctx_cat_np = np.zeros((B, Lmax), dtype=np.int64)
                ctx_mask_np = np.zeros((B, Lmax), dtype=bool)
                for i, (c, cc) in enumerate(zip(chunk_ctx, chunk_cat)):
                    L = len(c)
                    ctx_gidx_np[i, :L] = c
                    ctx_cat_np[i, :L] = cc
                    ctx_mask_np[i, :L] = True
                ctx_gidx_t = torch.tensor(ctx_gidx_np, device=device)
                ctx_cat_t = torch.tensor(ctx_cat_np, device=device)
                ctx_mask_t = torch.tensor(ctx_mask_np, device=device)

                P = cand_all_per_member[0].shape[0]
                dist_avg_sum = torch.zeros(B, P, device=device)

                for m, cand_all in zip(models, cand_all_per_member):
                    x_ctx = m.encode_feature(base_repr[ctx_gidx_t])
                    cat_s_flat = F.one_hot(ctx_cat_t.reshape(-1), C).float()
                    cat_t_flat = torch.zeros(B * Lmax, C, device=device)
                    cat_t_flat[:, cat_to_idx[cat]] = 1.0
                    f_ctx = m.embed_from_feature(x_ctx.reshape(B * Lmax, -1), cat_s_flat, cat_t_flat).reshape(B, Lmax, -1)

                    dist_sum = torch.zeros(B, P, device=device)
                    for c in range(C):
                        sel = (ctx_cat_t == c) & ctx_mask_t
                        if not sel.any():
                            continue
                        sims_c = torch.einsum("bld,pd->blp", f_ctx, cand_all[:, c, :])
                        dist_c = (2.0 - 2.0 * sims_c) * sel.unsqueeze(-1).float()
                        dist_sum += dist_c.sum(dim=1)
                    n_ctx = ctx_mask_t.sum(dim=1, keepdim=True).clamp(min=1).float()
                    dist_avg_sum += dist_sum / n_ctx

                dist_avg_ensemble = (dist_avg_sum / len(models)).cpu().numpy()
                tpos_arr = np.array(chunk_tpos)
                target_dist = dist_avg_ensemble[np.arange(B), tpos_arr]
                ranks = (dist_avg_ensemble <= target_dist[:, None]).sum(axis=1)

                n_total += B
                for k in ks:
                    hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped

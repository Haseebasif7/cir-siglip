"""
Phase 31: Modal GPU training app for the fair-baseline-strengthened
OutfitTransformer reproduction. Own app ("phase31-fair-baseline-ot"), own
`train_one` function; reuses the existing "phase27-text-category-data"
volume (Modal volumes are shared storage, not architecture-scoped -- it
already holds siglip_base.npz, text_embeddings.npz, cir_val_benchmark.json,
cir_test_benchmark.json; this phase additionally uploads training_data.json
and cir_train_benchmark.json, see 00_verify_volume_data.py). Every config
`name` is prefixed `ot31_` so files never collide with phases 27-30's own
checkpoints already on this volume.

What changed vs. week4/phase14b_outfittransformer_category_negatives, and why:

1. TEXT INPUT (step 1): `self.proj` (model.py) already took `siglip_dim` as
   a constructor argument -- no architecture code changed. `input_mode`
   selects base_repr = image (768-d) or normalize(concat(image,text))
   (1536-d), identical construction to phase 27/28/30's own text
   integration. Text enters through the SAME item-token pathway for every
   role (context, target, negatives, eval candidates, eval query context).

2. CHECKPOINT SELECTION FIX (step 2): phase 14b selected checkpoints by
   validation LOSS (train_core.py:291-300), never computing Recall@K during
   training at all. Verified directly against phase 14b's own
   training_curves.json: val_D_pos > val_D_neg at BOTH epoch 0 (1.2559 >
   1.0785) and epoch 91 (1.4059 > 1.4035) -- the triplet margin was never
   satisfied across the entire 92-epoch run, meaning val_loss's
   "improvement" was almost entirely the uniformity regularizer, not
   ranking quality. This phase adds a real evaluate_recall_gpu (below),
   evaluated every epoch against cir_val_benchmark.json, and selects/early-
   stops on val Recall@10 instead -- exactly phase 23's own fix for the
   project's own model. val_loss/val_D_pos/val_D_neg are still logged as
   secondary diagnostics (this is what supplies the "the two criteria
   disagree" evidence for checkpoint_selection_check.md).

3. BENCHMARK DISCIPLINE FIX: phase 14b evaluated directly against
   week4/phase12_controllable_modes/data/cir_benchmark.json -- the file
   every phase since 23 treats as the TEST set, touched exactly once.
   train_one NEVER loads cir_test_benchmark.json (structurally enforced,
   not just by convention -- grep this file, it does not appear). Only
   cir_val_benchmark.json (selection) and cir_train_benchmark.json
   (overfitting diagnostic, reused unchanged from week4/phase25_scale/,
   built from the identical Polyvore nondisjoint split training_data.json
   already uses) are loaded here.

4. VECTORIZED NEGATIVE SAMPLER: phase 14b's `_sample_negatives` (random
   mode) built `extra_pool = [i for i in pool[category] if i not in
   exclude]` per sample -- an O(pool size) Python list comprehension over a
   mean ~36,500-item category pool, measured at ~56s/epoch of pure sampling
   overhead. TrainState.sample_negatives below replaces this with an
   index-space rejection sampler operating on a precomputed numpy int64
   array per category, verified statistically equivalent to the original
   (00a_sampler_equivalence_check.py: 200,000-draw two-sample chi-square,
   stat/dof=1.005, p=0.265, zero exclude-set violations, ~103x speedup) --
   a speed-only refactor, the category-restricted-uniform-random SCHEME
   itself (phase 14b's own adopted fix) is unchanged. One minor, deliberate,
   documented deviation: this sampler suppresses within-call duplicate
   canonical items (the original's position-based `random.sample` could,
   at very low probability, draw the same item's value twice via two of
   its duplicate positions in the category multiset); negligible at this
   pool scale and arguably more correct for a triplet loss's negative set.
   NOTE the category lists ARE real multisets (an item appears once per
   outfit containing it, e.g. "shoes": 51,132 entries, 36,481 unique) -- the
   original algorithm samples proportional to outfit-occurrence count, and
   this vectorized version preserves that exactly (pool_gidx is built by
   mapping the RAW, duplicated per-category item list through the canonical
   id->index table, not by deduplicating first).

5. SEED PLUMBING: phase 14b never called torch.manual_seed, and TrainState
   was hardcoded to seed=0 -- its single reported run was not reproducible.
   Fixed here: torch.manual_seed(cfg["seed"]) before model construction,
   TrainState(seed=cfg["seed"]) for data order, np.random.default_rng
   (cfg["seed"]) for the vectorized sampler.

Everything else (transformer architecture shape, embed_query/embed_item_alone
split, triplet-margin + uniformity loss formulation, AdamW+OneCycleLR
schedule shape, category-restricted-random negative SCHEME) is unchanged
from phase 14b.
"""
import math
import time

import modal

app = modal.App("phase31-fair-baseline-ot")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy", "scipy")
volume = modal.Volume.from_name("phase27-text-category-data", create_if_missing=False)

DATA_DIR = "/data"
NUM_NEGATIVES = 10
UNIFORMITY_WEIGHT = 1.0  # step 3b sweeps this
MARGIN = 0.3             # step 3b sweeps this

# Phase 14b's exact original hyperparameters, held as the A1/A2/A3 measurement
# point (step 2) before any tuning (step 3) changes them.
DEFAULT_CONFIG = {
    "name": "default",
    "input_mode": "image",       # "image" (768-d) | "image_text" (1536-d, step 1)
    "d_model": 128, "d_embed": 64, "n_heads": 8, "n_layers": 4, "d_ffn": 512, "dropout": 0.1,
    "margin": MARGIN, "uniformity_weight": UNIFORMITY_WEIGHT,
    "lr": 2e-5, "batch_size": 96, "max_epochs": 100, "patience": 8,
    "min_delta": 0.0005,          # val-Recall@10 improvement threshold (step 2 fix; phase 14b had none, loss-based)
    "num_negatives": NUM_NEGATIVES,
    "seed": 42,
    "eval_every": 1,
    "eval_train_benchmark": True,  # step 4's overfitting diagnostic; cheap enough to leave on always
    "save_checkpoint": False,
    "selection_metric": "recall10",  # "recall10" (step 2 fix, default for every step 3+ run) | "val_loss"
                                      # (phase 14b's ORIGINAL criterion -- only used for the A1 fidelity
                                      # check, to isolate the selection-fix's own effect from the Modal
                                      # port itself before crediting it with anything in A2).
}


def _build_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class OutfitTransformerSigLIP(nn.Module):
        def __init__(self, siglip_dim, d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512, dropout=0.1):
            super().__init__()
            self.proj = nn.Linear(siglip_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
                dropout=dropout, batch_first=True, norm_first=True, activation=F.mish,
            )
            self.set_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
            self.embed_ffn = nn.Linear(d_model, d_embed, bias=False)
            self.outfit_token = nn.Parameter(torch.randn(d_model) * 0.02)

        def encode_item_tokens(self, siglip_vecs):
            x = F.normalize(siglip_vecs, p=2, dim=-1)
            e = self.proj(x)
            return F.normalize(e, p=2, dim=-1)

        def embed_query(self, ctx_tokens, pad_mask):
            B = ctx_tokens.shape[0]
            tok = self.outfit_token.view(1, 1, -1).expand(B, -1, -1)
            seq = torch.cat([tok, ctx_tokens], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device), pad_mask], dim=1)
            h = self.set_enc(seq, src_key_padding_mask=pad)
            out = self.embed_ffn(h[:, 0, :])
            return F.normalize(out, p=2, dim=-1)

        def embed_item_alone(self, item_tokens):
            seq = item_tokens.unsqueeze(1)
            h = self.set_enc(seq, src_key_padding_mask=None)
            out = self.embed_ffn(h[:, 0, :])
            return F.normalize(out, p=2, dim=-1)

    def uniformity_loss(f, t=2.0):
        sq_dists = torch.cdist(f, f, p=2) ** 2
        B = f.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=f.device)
        return torch.log(torch.exp(-t * sq_dists[mask]).mean())

    return OutfitTransformerSigLIP, uniformity_loss


@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=10800, max_containers=5)
def train_one(config: dict) -> dict:
    import json
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    cfg = {**DEFAULT_CONFIG, **config}
    assert cfg["name"].startswith("ot31_"), "every phase31 config name must be prefixed ot31_"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OutfitTransformerSigLIP, uniformity_loss = _build_model()
    data_dir = Path(DATA_DIR)

    # --- Credit-safety: idempotent skip + warm-start resume, unchanged
    # pattern from phase 27-30. ---
    status_path = data_dir / f"status_{cfg['name']}.json"
    ckpt_path = data_dir / f"checkpoint_{cfg['name']}.pt"

    if status_path.exists():
        with open(status_path) as f:
            prior_status = json.load(f)
        if prior_status.get("status") == "done":
            print(f"{cfg['name']}: already done (val_recall10="
                  f"{prior_status.get('best_recall10')}), skipping -- credit-safety short-circuit.")
            return prior_status["result"]

    torch.manual_seed(cfg["seed"])

    # --- Load base representation. NEVER loads cir_test_benchmark.json --
    # structurally enforced, see module docstring point 3. ---
    img_npz = np.load(data_dir / "siglip_base.npz", allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    n_items = len(item_ids)

    image_emb = img_npz["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    image_t = torch.tensor(image_emb, device=device)

    if cfg["input_mode"] == "image_text":
        text_npz = np.load(data_dir / "text_embeddings.npz", allow_pickle=True)
        text_ids = [str(a) for a in text_npz["item_ids"]]
        assert text_ids == item_ids, "text_embeddings.npz must be positionally aligned to siglip_base.npz"
        text_emb = text_npz["embeddings"].astype(np.float32)
        text_t = torch.tensor(text_emb, device=device)
        base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)  # (n_items, 1536)
        in_dim = 1536
    else:
        base_repr = image_t  # already normalized above; (n_items, 768)
        in_dim = 768

    # --- Training data: outfits + category pools, identical source to
    # phase 14b (week4/phase13_csa_net_baseline/data/training_data.json,
    # uploaded onto this volume by 00_verify_volume_data.py). ---
    with open(data_dir / "training_data.json") as f:
        td = json.load(f)
    train_outfits = [o for o in td["train_outfits"] if len(o["items"]) >= 2]
    val_outfits = [o for o in td["val_outfits"] if len(o["items"]) >= 2]
    train_items_by_cat = td["train_items_by_category"]
    val_items_by_cat = td["val_items_by_category"]

    class TrainState:
        """Vectorized replacement for phase 14b's TrainState -- see module
        docstring point 4 for the equivalence argument. category pool
        arrays are multisets (an item appears once per outfit containing
        it), built from the RAW per-category item lists so duplicate
        weighting (more outfit-occurrences = more likely to be drawn) is
        preserved exactly."""

        def __init__(self, items_by_cat, seed):
            self.rng_np = np.random.default_rng(seed)
            self.item_cat = {}
            self.cat_pool_gidx = {}
            for cat, items in items_by_cat.items():
                for i in items:
                    self.item_cat[i] = cat
                self.cat_pool_gidx[cat] = np.array(
                    [id_to_gidx[i] for i in items if i in id_to_gidx], dtype=np.int64
                )

        def sample_negatives(self, category, exclude_gidx_set, num_negatives):
            pool = self.cat_pool_gidx.get(category)
            if pool is None or len(pool) == 0:
                return np.array([], dtype=np.int64)
            n_pool = len(pool)
            k = min(num_negatives, max(n_pool - len(exclude_gidx_set), 0))
            if k <= 0:
                return np.array([], dtype=np.int64)
            chosen, seen = [], set()
            attempts = 0
            draw_size = min(n_pool, k + 8)
            while len(chosen) < k and attempts < 20:
                cand_pos = self.rng_np.integers(0, n_pool, size=draw_size)
                cand_gidx = pool[cand_pos]
                for g in cand_gidx:
                    gi = int(g)
                    if gi not in exclude_gidx_set and gi not in seen:
                        seen.add(gi)
                        chosen.append(gi)
                        if len(chosen) >= k:
                            break
                attempts += 1
            return np.array(chosen, dtype=np.int64)

        def make_sample(self, outfit_record, num_negatives):
            items = outfit_record["items"]
            target = items[int(self.rng_np.integers(0, len(items)))]
            context = [i for i in items if i != target]
            target_cat = self.item_cat[target]
            exclude_gidx = {id_to_gidx[i] for i in items if i in id_to_gidx}
            neg_gidx = self.sample_negatives(target_cat, exclude_gidx, num_negatives)
            return {
                "context_gidx": np.array([id_to_gidx[i] for i in context if i in id_to_gidx], dtype=np.int64),
                "target_gidx": id_to_gidx[target],
                "neg_gidx": neg_gidx,
            }

    def build_batch(samples, device):
        B = len(samples)
        lengths = [len(s["context_gidx"]) for s in samples]
        Lmax = max(lengths)
        ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
        ctx_mask = np.ones((B, Lmax), dtype=bool)
        for i, s in enumerate(samples):
            L = len(s["context_gidx"])
            ctx_gidx[i, :L] = s["context_gidx"]
            ctx_mask[i, :L] = False

        target_gidx = np.array([s["target_gidx"] for s in samples], dtype=np.int64)

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
            base_repr[torch.tensor(ctx_gidx, device=device)],           # (B, Lmax, in_dim)
            torch.tensor(ctx_mask, device=device),
            base_repr[torch.tensor(target_gidx, device=device)],        # (B, in_dim)
            base_repr[torch.tensor(neg_gidx, device=device)],           # (B, Mmax, in_dim)
            torch.tensor(neg_mask, device=device),
        )

    def category_negative_triplet_loss(query_emb, answer_emb, neg_emb, neg_mask, margin):
        pos = torch.norm(query_emb - answer_emb, p=2, dim=-1)
        d_negs = torch.norm(query_emb.unsqueeze(1) - neg_emb, p=2, dim=-1)
        d_negs = d_negs.masked_fill(~neg_mask, float("inf"))
        hardest_neg, _ = d_negs.min(dim=1)
        loss = F.relu(pos - hardest_neg + margin)
        return loss.mean(), pos.mean().item(), hardest_neg.mean().item()

    def compute_batch_loss(model, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, margin, uniformity_weight):
        B, Lmax, D = ctx_vecs.shape
        ctx_tokens = model.encode_item_tokens(ctx_vecs.reshape(B * Lmax, D)).reshape(B, Lmax, -1)
        query_emb = model.embed_query(ctx_tokens, ctx_mask)

        target_tokens = model.encode_item_tokens(target_vecs)
        answer_emb = model.embed_item_alone(target_tokens)

        Bn, M, Dn = neg_vecs.shape
        neg_tokens = model.encode_item_tokens(neg_vecs.reshape(Bn * M, Dn)).reshape(Bn, M, -1)
        neg_emb = model.embed_item_alone(neg_tokens.reshape(Bn * M, -1)).reshape(Bn, M, -1)

        triplet_loss, d_pos, d_neg = category_negative_triplet_loss(query_emb, answer_emb, neg_emb, neg_mask, margin)
        total_loss = triplet_loss
        if uniformity_weight > 0:
            total_loss = total_loss + uniformity_weight * uniformity_loss(answer_emb)
        return total_loss, d_pos, d_neg

    # --- Evaluator: adapted from phase 14b's 04b_cir_eval_random_negatives.py
    # (embed_query/embed_item_alone split), made GPU-resident and batched.
    # bench_key selects which file to read -- "val" (selection signal, every
    # epoch) or "train" (overfitting diagnostic, step 4). Never "test". ---
    QUERY_BATCH = 1024
    _bench_cache = {}

    def load_bench(bench_key):
        if bench_key not in _bench_cache:
            fname = "cir_val_benchmark.json" if bench_key == "val" else "cir_train_benchmark.json"
            with open(data_dir / fname) as f:
                b = json.load(f)
            _bench_cache[bench_key] = (b["pools"], b["queries"])
        return _bench_cache[bench_key]

    def evaluate_recall_gpu(model, bench_key, ks=(10, 30, 50)):
        pools, queries = load_bench(bench_key)
        model.eval()
        with torch.no_grad():
            item_tokens_all = model.encode_item_tokens(base_repr)  # (n_items, d_model)
            cand_emb_all = model.embed_item_alone(item_tokens_all).cpu().numpy()  # (n_items, d_embed)

            by_cat = defaultdict(list)
            for qi, q in enumerate(queries):
                by_cat[q["category"]].append(qi)

            hits = {k: 0 for k in ks}
            n_total, n_skipped = 0, 0
            for cat, qidxs in by_cat.items():
                if cat not in pools:
                    n_skipped += len(qidxs)
                    continue
                pool_ids = pools[cat]
                pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
                pool_idx = [id_to_gidx[i] for i in pool_ids if i in id_to_gidx]
                pool_emb = cand_emb_all[pool_idx]

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
                    q_emb = model.embed_query(tokens, mask_t)
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

    # --- Model + optimizer ---
    model = OutfitTransformerSigLIP(
        siglip_dim=in_dim, d_model=cfg["d_model"], d_embed=cfg["d_embed"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ffn=cfg["d_ffn"], dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    steps_per_epoch = max(1, math.ceil(len(train_outfits) / cfg["batch_size"]))
    total_steps = cfg["max_epochs"] * steps_per_epoch + 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )

    state = TrainState(train_items_by_cat, seed=cfg["seed"])
    val_state = TrainState(val_items_by_cat, seed=42)  # fixed seed, matches phase 14b's own fixed val-sample convention
    val_samples_fixed = [val_state.make_sample(o, cfg["num_negatives"]) for o in val_outfits]

    best_recall10 = -1.0
    best_selection_score = -1.0   # used when selection_metric == "recall10"
    best_val_loss_seen = float("inf")  # used when selection_metric == "val_loss"
    best_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    if ckpt_path.exists():
        print(f"{cfg['name']}: found an existing checkpoint, warm-starting from it "
              f"(optimizer/epoch state is NOT restored, only weights).")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if status_path.exists():
            with open(status_path) as f:
                prior_status = json.load(f)
            best_recall10 = prior_status.get("best_recall10", -1.0) or -1.0
            print(f"  warm-started at best_recall10={best_recall10}")

    for epoch in range(cfg["max_epochs"]):
        model.train()
        order = list(range(len(train_outfits)))
        state.rng_np.shuffle(order)
        epoch_losses, epoch_d_pos, epoch_d_neg = [], [], []
        for b_start in range(0, len(order), cfg["batch_size"]):
            b_idx = order[b_start:b_start + cfg["batch_size"]]
            if len(b_idx) < 2:
                continue
            samples = [state.make_sample(train_outfits[i], cfg["num_negatives"]) for i in b_idx]
            ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask = build_batch(samples, device)

            optimizer.zero_grad()
            loss, d_pos, d_neg = compute_batch_loss(
                model, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, cfg["margin"], cfg["uniformity_weight"]
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            epoch_d_pos.append(d_pos)
            epoch_d_neg.append(d_neg)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Secondary diagnostics: val_loss/D_pos/D_neg (the OLD selection
        # signal, kept only for checkpoint_selection_check.md's evidence).
        model.eval()
        val_losses, val_d_pos, val_d_neg = [], [], []
        with torch.no_grad():
            for b_start in range(0, len(val_samples_fixed), cfg["batch_size"]):
                batch = val_samples_fixed[b_start:b_start + cfg["batch_size"]]
                if len(batch) < 2:
                    continue
                ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask = build_batch(batch, device)
                loss, d_pos, d_neg = compute_batch_loss(
                    model, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, cfg["margin"], cfg["uniformity_weight"]
                )
                val_losses.append(loss.item())
                val_d_pos.append(d_pos)
                val_d_neg.append(d_neg)
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        do_eval = (epoch % cfg["eval_every"] == 0) or (epoch == cfg["max_epochs"] - 1)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_D_pos": float(np.mean(val_d_pos)) if val_d_pos else 0.0,
               "val_D_neg": float(np.mean(val_d_neg)) if val_d_neg else 0.0}
        if do_eval:
            # Recall is ALWAYS computed here regardless of selection_metric --
            # this is what lets checkpoint_selection_check.md show where val
            # Recall@10 actually peaks vs. where val_loss bottoms, even for
            # the A1 run that selects on val_loss (phase 14b's original
            # criterion, kept only for that one fidelity-check run).
            recall, n_total, n_skipped = evaluate_recall_gpu(model, "val")
            row.update({"recall10": recall[10], "recall30": recall[30], "recall50": recall[50]})
            if cfg["eval_train_benchmark"]:
                train_recall, _, _ = evaluate_recall_gpu(model, "train")
                row.update({"train_recall10": train_recall[10], "train_recall30": train_recall[30]})

            if cfg["selection_metric"] == "val_loss":
                # phase 14b's ORIGINAL criterion (train_core.py:291-300),
                # ported unchanged: lower is better, threshold 1e-4 absolute
                # (not cfg["min_delta"], which is the new recall-based
                # threshold and has a different natural scale).
                selection_score = -val_loss
                improved = val_loss < best_val_loss_seen - 1e-4
                best_val_loss_seen = min(best_val_loss_seen, val_loss)
            else:
                selection_score = recall[10]
                improved = selection_score > best_selection_score + cfg["min_delta"]

            if improved:
                best_selection_score = selection_score
                best_recall10 = recall[10]  # always reported in terms of recall10, whichever metric selected it
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
                if cfg.get("save_checkpoint"):
                    torch.save(best_state, ckpt_path)
                    with open(status_path, "w") as f:
                        json.dump({"status": "in_progress", "best_epoch": best_epoch,
                                    "best_recall10": best_recall10, "epoch_reached": epoch}, f)
                    volume.commit()
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    curve.append(row)
                    break
        curve.append(row)

    wall_time = time.time() - t_start

    result = {
        "config": cfg,
        "curve": curve,
        "best_epoch": best_epoch,
        "best_recall10": best_recall10,
        "n_epochs_run": len(curve),
        "wall_time_sec": wall_time,
        "device": device,
        "in_dim": in_dim,
        "n_params": n_params,
    }

    if cfg.get("save_checkpoint") and best_state is not None:
        torch.save(best_state, ckpt_path)
        result["checkpoint_path"] = str(ckpt_path)
        with open(status_path, "w") as f:
            json.dump({"status": "done", "best_epoch": best_epoch,
                        "best_recall10": best_recall10, "result": result}, f)
        volume.commit()

    return result


@app.local_entrypoint()
def smoke_test():
    """`modal run scripts/modal_app.py::smoke_test` -- 2-epoch sanity check
    (image-only and image+text) before committing to any real run. Also
    reports measured s/epoch to calibrate the step 3/4 grid budgets."""
    import json
    for input_mode in ("image", "image_text"):
        cfg = {"name": f"ot31_smoke_{input_mode}", "input_mode": input_mode,
               "max_epochs": 2, "patience": 5, "eval_every": 1, "save_checkpoint": False}
        t0 = time.time()
        result = train_one.remote(cfg)
        wall = time.time() - t0
        print(f"\n=== {input_mode} ===")
        print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
        print(f"measured s/epoch (incl. Modal overhead): {wall / result['n_epochs_run']:.1f}")
        print("curve:")
        for row in result["curve"]:
            print(" ", row)

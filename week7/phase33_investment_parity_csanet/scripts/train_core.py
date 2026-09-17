"""
Phase 33: vectorized training/eval core for CSA-Net-on-SigLIP, replacing
phase 13b's per-sample, per-negative, per-context-item Python-loop
computation with batched tensor operations that call the SAME unchanged
model.py methods (see model.py's own docstring on why embed_from_feature
stays 2D-only, flattened into by this file).

WHY THIS MATTERS: phase 13b's original 40-epoch run took ~6.9 hours on a
local M4 Air (24,872s, `week4/phase13b_csa_net_siglip_backbone/
training_log.md`). Its compute_batch_loss does one `model.embed_from_feature`
call PER NEGATIVE PER SAMPLE (10 negatives x 96-sample batch = 960 tiny
forward calls per training step, ~555 steps/epoch) -- the exact class of
MPS/CPU per-op dispatch overhead this project's own PROJECT_MEMORY already
documents (phase 15's finding: many small ops can be far slower than a few
large batched ones, independent of raw FLOPs). Verified equivalent to the
original scalar computation on synthetic data before any real training run
-- see 00_verify_vectorization.py.

CATEGORY-SHARING ASSUMPTION (carried over unchanged from phase 13b, not
introduced here): negatives are mined same-category candidates for the
positive item (`neg_candidates[split][positive_id]`), so they share the
positive's category by construction -- `cat_t` (target/positive category)
is one shared value per sample, used for both the positive AND every
negative's conditioning. This is what makes the vectorization tractable
(one cat_t per sample, broadcast, rather than a per-negative category).
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
    containing it) -- category assignment (item_cat) and negative mining are
    otherwise identical to phase 13b's own TrainState, just seeded via
    numpy for consistency with phase 31/32's own seed-plumbing fix (phase
    13b, like phase 14b, never called torch.manual_seed or seeded its own
    RNG beyond a hardcoded default -- fixed here the same way)."""

    def __init__(self, id_to_gidx, item_cat, cat_to_idx, train_items_by_cat, val_items_by_cat,
                 neg_candidates, split_key, seed):
        self.rng = random.Random(seed)
        self.id_to_gidx = id_to_gidx
        self.item_cat = item_cat
        self.cat_to_idx = cat_to_idx
        self.items_by_cat = train_items_by_cat if split_key == "train" else val_items_by_cat
        self.neg_candidates = neg_candidates[split_key]
        self.split_key = split_key

    def _sample_negatives(self, positive_id, category, exclude, num_negatives):
        cands = self.neg_candidates.get(positive_id, [])
        cands = [c for c in cands if c not in exclude]
        if len(cands) >= num_negatives:
            return self.rng.sample(cands, num_negatives)
        extra_pool = [i for i in self.items_by_cat[category] if i not in exclude and i not in cands]
        n_extra = num_negatives - len(cands)
        extra = self.rng.sample(extra_pool, min(n_extra, len(extra_pool))) if extra_pool else []
        return cands + extra

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
    downstream of this function."""
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
    """Vectorized equivalent of phase 13b's per-sample/per-negative loop.
    Reproduces the EXACT aggregation order: mean over context items happens
    per (sample, negative) BEFORE min-over-negatives, matching the original
    (verified in 00_verify_vectorization.py). See module docstring for why
    a single shared cat_t per sample is valid (mined negatives share the
    positive's category)."""
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
    """Vectorized CIR evaluator. Groups context items by their OWN category
    (only num_categories=11 possible values) and does one dense einsum per
    category against the precomputed candidate table, instead of phase 13b's
    original one-tiny-forward-call-per-context-item-per-query loop -- avoids
    both the Python-loop overhead AND a naive per-position candidate-table
    gather that would blow up memory (B*Lmax*pool_size*embed_dim)."""
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


def load_catalog(embeddings_npz, training_data_path, negative_candidates_path,
                  text_embeddings_npz=None, metadata_path=None, device="cpu"):
    """Loads everything shared across train/val TrainStates: base_repr
    (image-only 768-d, or image+text 1536-d if text_embeddings_npz given --
    identical construction to phase 27/28/31/32), category tables, and
    mined negative candidates. Returns a dict of everything run_training and
    the standalone eval/tuning scripts need, so this loading logic isn't
    duplicated across every driver script."""
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
    # test-benchmark query-context items) -- exactly phase 13b's own
    # 03_csa_cir_eval.py precedent: look up semantic_category from the raw
    # Polyvore item metadata for anything item_cat doesn't already cover.
    # Skipped by default (metadata_path=None) since steps 1-4 never hit this
    # gap (val benchmark + training data stay within the covered universe);
    # only the final test-benchmark eval needs it -- see 06_final_test_eval.py.
    if metadata_path is not None:
        with open(metadata_path) as f:
            meta = json.load(f)
        for item_id, v in meta.items():
            cat = v.get("semantic_category")
            if item_id not in item_cat and cat in categories:
                item_cat[item_id] = cat

    with open(negative_candidates_path) as f:
        neg_cands_raw = json.load(f)
    neg_candidates = {"train": neg_cands_raw.get("train", {}), "val": neg_cands_raw.get("val", {})}

    return {
        "base_repr": base_repr, "in_dim": in_dim, "id_to_gidx": id_to_gidx,
        "categories": categories, "cat_to_idx": cat_to_idx, "item_cat": item_cat,
        "train_outfits": train_outfits, "val_outfits": val_outfits,
        "train_items_by_cat": train_items_by_cat, "val_items_by_cat": val_items_by_cat,
        "neg_candidates": neg_candidates,
    }


def run_training(catalog, device, max_epochs, batch_size, lr, patience, margin=MARGIN,
                  uniformity_weight=UNIFORMITY_WEIGHT, num_negatives=NUM_NEGATIVES,
                  seed=42, min_delta=0.0005, selection_metric="recall10", eval_every=1,
                  val_benchmark=None, log_every=100):
    """selection_metric: "recall10" (default, this phase's fix) | "val_loss"
    (phase 13b's ORIGINAL criterion -- only for the one fidelity-check run).
    val_benchmark: (pools, queries) tuple, required when selection_metric=
    "recall10" or for logging recall alongside a val_loss-selected run (kept
    optional so the pure fidelity-check run can skip it if truly not needed,
    though in practice this phase always evaluates recall for the disagreement
    evidence -- see checkpoint_selection_check.md)."""
    torch.manual_seed(seed)
    base_repr = catalog["base_repr"]
    id_to_gidx = catalog["id_to_gidx"]
    cat_to_idx = catalog["cat_to_idx"]
    item_cat = catalog["item_cat"]
    num_categories = len(cat_to_idx)

    train_state = TrainState(id_to_gidx, item_cat, cat_to_idx, catalog["train_items_by_cat"],
                              catalog["val_items_by_cat"], catalog["neg_candidates"], "train", seed=seed)
    val_state = TrainState(id_to_gidx, item_cat, cat_to_idx, catalog["train_items_by_cat"],
                            catalog["val_items_by_cat"], catalog["neg_candidates"], "val", seed=42)
    val_samples_fixed = [val_state.make_sample(o, num_negatives) for o in catalog["val_outfits"]]

    model = CSANetSigLIP(num_categories=num_categories, siglip_dim=catalog["in_dim"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(catalog["train_outfits"]) // batch_size)
    total_steps = max_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    curve = []
    best_selection_score = -1.0
    best_val_loss_seen = float("inf")
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

            if selection_metric == "val_loss":
                improved = val_loss < best_val_loss_seen - 1e-4
                best_val_loss_seen = min(best_val_loss_seen, val_loss)
            else:
                improved = recall[10] > best_selection_score + min_delta

            if improved:
                if selection_metric != "val_loss":
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
        elif do_eval:
            # selection_metric == "val_loss" and no val_benchmark given -- pure loss-based fidelity check
            improved = val_loss < best_val_loss_seen - 1e-4
            if improved:
                best_val_loss_seen = val_loss
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
    PRINCIPLE as phases 26/28/30/32, just applied to the score this
    architecture's own ranking is based on. Never averages embeddings --
    independently-trained models share no coordinate system. A single-
    element `models` list reproduces solo evaluate_recall."""
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

"""
Phase 15b: training loop with alpha FULLY REMOVED from the loss-weighting
envelope (phase 15's diagnosed "alpha double-duty confound" fix).

Per training step: build the SAME batch of anchors as phase 15 (complement
sub-batch: CSA-Net leave-one-out outfit samples with mined same-category
negatives; substitute sub-batch: each complement sample's positive item as
ranking-distillation anchor, looked up against same-category-filtered SigLIP
neighbors). Unlike phase 15, there is no sampled alpha and no envelope:

    complement_loss  is ALWAYS computed with alpha_forward=0.0
    substitute_loss  is ALWAYS computed with alpha_forward=1.0
    total_loss = complement_loss + weight_sub * substitute_loss
               + UNIFORMITY_WEIGHT * uniformity_loss(representative_embeddings)

Both objectives are trained on EVERY step, simultaneously, with fixed
weights -- alpha is now a pure conditioning input with zero influence on
training dynamics, directly targeting phase 15's diagnosed confound (alpha
serving as both the attn_net input and the outer loss weight, letting the
network satisfy the loss at either endpoint without attn_net ever needing to
use alpha meaningfully).

`weight_sub` is REUSED from phase 15's own calibration (9.5801,
`week4/phase15_controllable_subspace_attention/loss_balancing_check.md`) --
not re-derived, since the underlying loss formulations are unchanged, only
how alpha interacts with them (per this phase's own brief).

Data plumbing (TrainState, build_batch_vectors, build_substitute_pairs,
make_batch) is copied UNCHANGED from phase 15's train_core.py.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import (
    CSANetSigLIPControllable,
    pairwise_distance,
    outfit_ranking_loss,
    uniformity_loss,
    ranking_distillation_loss,
)

NUM_NEGATIVES = 10  # matches phase 13/13b/15's choice, for direct comparability
UNIFORMITY_WEIGHT = 1.0  # matches phase 13/13b/15's choice
SUBSTITUTE_K = 20  # matches phase 15's choice
MIN_NEIGHBORS = 5  # matches phase 15's 00_build_same_category_neighbors.py threshold
TAU_DISTILL = 0.07  # matches phase 12c/15's ranking-distillation temperature
WEIGHT_SUB = 9.5801  # REUSED from phase 15's loss_balancing_check.md, not re-derived


class TrainState:
    """Copied unchanged from phase 15's train_core.py."""

    def __init__(self, embeddings_npz, training_data_path, negative_candidates_path,
                 same_category_neighbors_npz, seed=0):
        self.rng = random.Random(seed)

        data = np.load(embeddings_npz, allow_pickle=True)
        item_ids = [str(a) for a in data["item_ids"]]
        embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        self.idx = {a: i for i, a in enumerate(item_ids)}
        self.embeddings = embeddings  # (N, 768), L2-normalized

        nbr = np.load(same_category_neighbors_npz, allow_pickle=True)
        nbr_item_ids = [str(a) for a in nbr["item_ids"]]
        assert nbr_item_ids == item_ids, (
            "same_category_neighbors.npz ordering must match embeddings_npz ordering"
        )
        self.nbr_indices = nbr["indices"]  # (N, 50) int32, -1 padded
        self.nbr_sims = nbr["sims"]  # (N, 50) float32
        self.nbr_counts = nbr["counts"]  # (N,) int32

        with open(training_data_path) as f:
            td = json.load(f)
        self.categories = td["categories"]
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.train_outfits = td["train_outfits"]
        self.val_outfits = td["val_outfits"]
        self.train_items_by_cat = td["train_items_by_category"]
        self.val_items_by_cat = td["val_items_by_category"]

        self.item_cat = {}
        for cat, items in self.train_items_by_cat.items():
            for i in items:
                self.item_cat[i] = cat
        for cat, items in self.val_items_by_cat.items():
            for i in items:
                self.item_cat[i] = cat

        with open(negative_candidates_path) as f:
            neg_cands = json.load(f)
        self.neg_candidates = {"train": neg_cands.get("train", {}), "val": neg_cands.get("val", {})}

    def _sample_negatives(self, positive_id, category, split, exclude, num_negatives):
        cands = self.neg_candidates[split].get(positive_id, [])
        cands = [c for c in cands if c not in exclude]
        if len(cands) >= num_negatives:
            return self.rng.sample(cands, num_negatives)
        pool = self.train_items_by_cat if split == "train" else self.val_items_by_cat
        extra_pool = [i for i in pool[category] if i not in exclude and i not in cands]
        n_extra = num_negatives - len(cands)
        extra = self.rng.sample(extra_pool, min(n_extra, len(extra_pool))) if extra_pool else []
        return cands + extra

    def make_sample(self, outfit_record, split, num_negatives=NUM_NEGATIVES):
        items = outfit_record["items"]
        positive = self.rng.choice(items)
        context = [i for i in items if i != positive]
        pos_cat = self.item_cat[positive]
        exclude = set(items)
        negatives = self._sample_negatives(positive, pos_cat, split, exclude, num_negatives)
        return {
            "context_items": context,
            "context_cats": [self.item_cat[i] for i in context],
            "positive": positive,
            "positive_cat": pos_cat,
            "negatives": negatives,
        }

    def onehot(self, cat_name, batch=1):
        v = torch.zeros(batch, len(self.categories))
        v[:, self.cat_to_idx[cat_name]] = 1.0
        return v

    def substitute_pair_for(self, item_id, k=SUBSTITUTE_K):
        gidx = self.idx.get(item_id)
        if gidx is None:
            return None, None
        n_avail = int(self.nbr_counts[gidx])
        if n_avail < MIN_NEIGHBORS:
            return None, None
        n_use = min(n_avail, k)
        nbr_gidx = self.nbr_indices[gidx, :n_use]
        nbr_sims = self.nbr_sims[gidx, :n_use]
        return nbr_gidx, nbr_sims


def build_batch_vectors(state, samples, substitute_pairs, device):
    """Copied unchanged from phase 15's train_core.py."""
    unique_gidx = []
    seen = set()

    def add(gidx):
        if gidx not in seen:
            seen.add(gidx)
            unique_gidx.append(gidx)

    for s in samples:
        for i in s["context_items"] + [s["positive"]] + s["negatives"]:
            add(state.idx[i])
    for pair in substitute_pairs:
        if pair[0] is None:
            continue
        anchor_gidx, nbr_gidx, _ = pair
        add(anchor_gidx)
        for g in nbr_gidx:
            add(int(g))

    vecs = torch.tensor(state.embeddings[unique_gidx], device=device)
    gidx_to_pos = {g: p for p, g in enumerate(unique_gidx)}
    return vecs, gidx_to_pos


def state_item_id_for(state, gidx):
    if not hasattr(state, "_rev_idx"):
        state._rev_idx = {g: iid for iid, g in state.idx.items()}
    return state._rev_idx[gidx]


def build_substitute_pairs(state, samples):
    pairs = []
    for s in samples:
        nbr_gidx, teacher_sims = state.substitute_pair_for(s["positive"])
        if nbr_gidx is None:
            pairs.append((None, None, None))
        else:
            pairs.append((state.idx[s["positive"]], nbr_gidx, teacher_sims))
    return pairs


def make_batch(state, outfits, batch_ids, split, device):
    samples = [state.make_sample(outfits[i], split) for i in batch_ids]
    substitute_pairs = build_substitute_pairs(state, samples)
    vecs, gidx_to_pos = build_batch_vectors(state, samples, substitute_pairs, device)
    return samples, substitute_pairs, vecs, gidx_to_pos


def compute_batch_loss_decoupled(model, state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                                  weight_sub, aggregation="min", uniformity_weight=UNIFORMITY_WEIGHT):
    """The core change this phase: complement ALWAYS at alpha=0, substitute
    ALWAYS at alpha=1, both computed and backpropagated every step, combined
    with fixed weights (no alpha-dependent envelope at all)."""
    ALPHA_COMP = 0.0
    ALPHA_SUB = 1.0

    # ---- complement sub-loss, fixed at alpha=0 ----
    comp_losses = []
    d_pos_list, d_neg_list = [], []
    representative_embeddings = []
    for s in samples:
        n_ctx = len(s["context_items"])
        if n_ctx == 0:
            continue
        ctx_pos = [gidx_to_pos[state.idx[i]] for i in s["context_items"]]
        x_ctx = x_all[ctx_pos]
        cat_s_ctx = torch.cat([state.onehot(c) for c in s["context_cats"]], dim=0).to(device)
        cat_t = state.onehot(s["positive_cat"], batch=n_ctx).to(device)
        f_ctx = model.embed_from_feature(x_ctx, cat_s_ctx, cat_t, ALPHA_COMP)

        x_pos = x_all[gidx_to_pos[state.idx[s["positive"]]]].unsqueeze(0).expand(n_ctx, -1)
        f_pos = model.embed_from_feature(x_pos, cat_s_ctx, cat_t, ALPHA_COMP)
        d_pos_i = pairwise_distance(f_ctx, f_pos)
        D_pos = d_pos_i.mean()
        representative_embeddings.append(F.normalize(f_pos.mean(dim=0), p=2, dim=-1))

        d_negs = []
        for neg_id in s["negatives"]:
            x_neg = x_all[gidx_to_pos[state.idx[neg_id]]].unsqueeze(0).expand(n_ctx, -1)
            f_neg = model.embed_from_feature(x_neg, cat_s_ctx, cat_t, ALPHA_COMP)
            d_neg_i = pairwise_distance(f_ctx, f_neg)
            d_negs.append(d_neg_i.mean())
        D_negs = torch.stack(d_negs)

        comp_losses.append(outfit_ranking_loss(D_pos.unsqueeze(0), D_negs.unsqueeze(0), aggregation=aggregation))
        d_pos_list.append(D_pos.item())
        d_neg_list.append(D_negs.min().item() if aggregation == "min" else D_negs.mean().item())

    complement_loss = torch.stack(comp_losses).mean() if comp_losses else torch.tensor(0.0, device=device)

    # ---- substitute sub-loss, fixed at alpha=1 ----
    sub_losses = []
    for pair in substitute_pairs:
        if pair[0] is None:
            continue
        anchor_gidx, nbr_gidx, teacher_sims = pair
        anchor_cat = state.item_cat.get(state_item_id_for(state, anchor_gidx))
        cat_pair = state.onehot(anchor_cat).to(device)
        n_nbr = len(nbr_gidx)

        x_anchor = x_all[gidx_to_pos[anchor_gidx]].unsqueeze(0)
        x_nbrs = x_all[[gidx_to_pos[int(g)] for g in nbr_gidx]]

        z_anchor = model.embed_from_feature(x_anchor, cat_pair, cat_pair, ALPHA_SUB)
        cat_pair_rep = cat_pair.expand(n_nbr, -1)
        z_nbrs = model.embed_from_feature(x_nbrs, cat_pair_rep, cat_pair_rep, ALPHA_SUB)

        teacher = torch.tensor(teacher_sims, device=device).unsqueeze(0)
        sub_losses.append(ranking_distillation_loss(z_anchor, z_nbrs.unsqueeze(0), teacher, tau=TAU_DISTILL))
        representative_embeddings.append(z_anchor.squeeze(0))

    substitute_loss = torch.stack(sub_losses).mean() if sub_losses else torch.tensor(0.0, device=device)

    # No alpha envelope: fixed weights, every step, both terms simultaneously.
    total_loss = complement_loss + weight_sub * substitute_loss
    if representative_embeddings and uniformity_weight > 0:
        total_loss = total_loss + uniformity_weight * uniformity_loss(torch.stack(representative_embeddings))

    d_pos_mean = float(np.mean(d_pos_list)) if d_pos_list else float("nan")
    d_neg_mean = float(np.mean(d_neg_list)) if d_neg_list else float("nan")
    return total_loss, d_pos_mean, d_neg_mean, complement_loss.item(), substitute_loss.item()


def run_training_decoupled(embeddings_npz, training_data_path, negative_candidates_path,
                            same_category_neighbors_npz, out_dir, device, max_epochs, batch_size, lr,
                            patience, weight_sub=WEIGHT_SUB, n_train_outfits=None, n_val_outfits=None,
                            log_every=200, uniformity_weight=UNIFORMITY_WEIGHT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "csa_net_decoupled_best.pt"

    state = TrainState(embeddings_npz, training_data_path, negative_candidates_path,
                        same_category_neighbors_npz, seed=0)
    train_outfits = state.train_outfits[:n_train_outfits] if n_train_outfits else state.train_outfits
    val_outfits = state.val_outfits[:n_val_outfits] if n_val_outfits else state.val_outfits

    model = CSANetSigLIPControllable().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = max_epochs * max(1, len(train_outfits) // batch_size)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    val_state = TrainState(embeddings_npz, training_data_path, negative_candidates_path,
                            same_category_neighbors_npz, seed=42)

    history = []
    best_val_loss = float("inf")
    epochs_since_improve = 0
    step = 0

    for epoch in range(max_epochs):
        model.train()
        order = list(range(len(train_outfits)))
        state.rng.shuffle(order)
        epoch_losses = []
        for b_start in range(0, len(order), batch_size):
            b_idx = order[b_start:b_start + batch_size]
            if len(b_idx) < 2:
                continue
            samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
                state, train_outfits, b_idx, "train", device)

            optimizer.zero_grad()
            x_all = model.encode_feature(vecs)
            loss, d_pos, d_neg, comp_l, sub_l = compute_batch_loss_decoupled(
                model, state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                weight_sub=weight_sub, uniformity_weight=uniformity_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f} "
                      f"D_pos={d_pos:.4f} D_neg={d_neg:.4f} comp={comp_l:.4f} sub={sub_l:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        model.eval()
        val_losses, val_d_pos, val_d_neg, val_comp, val_sub = [], [], [], [], []
        with torch.no_grad():
            val_order = list(range(len(val_outfits)))
            for b_start in range(0, len(val_order), batch_size):
                b_idx = val_order[b_start:b_start + batch_size]
                if len(b_idx) < 2:
                    continue
                samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
                    val_state, val_outfits, b_idx, "val", device)
                x_all = model.encode_feature(vecs)
                loss, d_pos, d_neg, comp_l, sub_l = compute_batch_loss_decoupled(
                    model, val_state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                    weight_sub=weight_sub, uniformity_weight=uniformity_weight)
                val_losses.append(loss.item())
                val_d_pos.append(d_pos)
                val_d_neg.append(d_neg)
                val_comp.append(comp_l)
                val_sub.append(sub_l)
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(epoch_losses))
        print(f"== epoch {epoch} done (decoupled): train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_D_pos={np.mean(val_d_pos):.4f} val_D_neg={np.mean(val_d_neg):.4f} "
              f"val_comp={np.mean(val_comp):.4f} val_sub={np.mean(val_sub):.4f} ==", flush=True)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_D_pos": float(np.mean(val_d_pos)), "val_D_neg": float(np.mean(val_d_neg)),
            "val_complement_loss": float(np.mean(val_comp)), "val_substitute_loss": float(np.mean(val_sub)),
        })
        with open(out_dir / "training_curves_decoupled.json", "w") as f:
            json.dump(history, f)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), out_dir / ckpt_name)
            print(f"  -> new best val_loss {val_loss:.4f}, checkpoint saved", flush=True)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
                break

    return history

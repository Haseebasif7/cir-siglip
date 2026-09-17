"""
Phase 15: training loop for control-conditioned CSA-Net subspace attention.

Per training step: sample one alpha ~ Uniform(0,1) (design decision 3),
build a complement sub-batch (CSA-Net leave-one-out outfit samples, same
convention as phase 13/13b, mined same-category negatives) and a substitute
sub-batch that reuses each complement sample's positive item as the
ranking-distillation anchor (design decision 3), looked up against its
same-category-filtered SigLIP neighbors (00_build_same_category_neighbors.py
output, design decision 2). Both losses are computed at the SAME forward-pass
alpha and combined as:

    total_loss = alpha * weight_sub * substitute_loss
               + (1 - alpha) * complement_loss
               + UNIFORMITY_WEIGHT * uniformity_loss(representative_embeddings)

`weight_sub` is calibrated once at alpha=0.5 (design decision 6) -- see
02_calibrate_and_verify.py / loss_balancing_check.md.

Conditioning-consistency safeguard (design decision 5, the "alpha
double-duty confound"): `compute_batch_loss` takes `alpha_forward` and
`alpha_weight` as SEPARATE arguments. Normal training steps pass the same
sampled value for both. The smoke test (01_smoke_test.py) additionally runs
a fraction of steps with `alpha_forward != alpha_weight`, decoupling "does
attn_net respond to alpha" from "does this step's loss envelope reward that
response" -- this is what keeps attn_net's conditioning pathway live near
the alpha=0/1 envelope endpoints during the smoke test's own steps, and is
checked directly via `model.attention_weight_shift`, not assumed fixed by
decoupling alone.
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

NUM_NEGATIVES = 10  # matches phase 13/13b's choice, for direct comparability
UNIFORMITY_WEIGHT = 1.0  # matches phase 13/13b's choice
SUBSTITUTE_K = 20  # same-category neighbors used per anchor (capped by availability)
MIN_NEIGHBORS = 5  # matches 00_build_same_category_neighbors.py's threshold
TAU_DISTILL = 0.07  # matches phase 12c's ranking-distillation temperature


class TrainState:
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
            "same_category_neighbors.npz ordering must match embeddings_npz ordering "
            "(both derive from phase 9's siglip_base.npz catalog order)"
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
        """Returns (neighbor_ids, teacher_sims) for item_id's same-category
        neighbors, or (None, None) if below MIN_NEIGHBORS. Design decision 2:
        cat_s = cat_t = item_id's own category for both anchor and neighbors."""
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
    """Dedup all unique global indices needed (complement context/positive/negatives
    plus substitute anchors/neighbors), look up once into state.embeddings."""
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

    vecs = torch.tensor(state.embeddings[unique_gidx], device=device)  # (N_unique, 768)
    gidx_to_pos = {g: p for p, g in enumerate(unique_gidx)}
    return vecs, gidx_to_pos


def compute_batch_loss(model, state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                        alpha_forward, alpha_weight, weight_sub, aggregation="min",
                        uniformity_weight=UNIFORMITY_WEIGHT):
    """Computes complement (CSA-Net outfit ranking) loss and substitute
    (ranking-distillation) loss at `alpha_forward` (fed into attn_net), then
    combines them weighted by `alpha_weight`'s envelope (see module docstring
    for why these are kept as separate arguments)."""
    # ---- complement sub-loss (CSA-Net leave-one-out outfit ranking) ----
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
        f_ctx = model.embed_from_feature(x_ctx, cat_s_ctx, cat_t, alpha_forward)

        x_pos = x_all[gidx_to_pos[state.idx[s["positive"]]]].unsqueeze(0).expand(n_ctx, -1)
        f_pos = model.embed_from_feature(x_pos, cat_s_ctx, cat_t, alpha_forward)
        d_pos_i = pairwise_distance(f_ctx, f_pos)
        D_pos = d_pos_i.mean()
        representative_embeddings.append(F.normalize(f_pos.mean(dim=0), p=2, dim=-1))

        d_negs = []
        for neg_id in s["negatives"]:
            x_neg = x_all[gidx_to_pos[state.idx[neg_id]]].unsqueeze(0).expand(n_ctx, -1)
            f_neg = model.embed_from_feature(x_neg, cat_s_ctx, cat_t, alpha_forward)
            d_neg_i = pairwise_distance(f_ctx, f_neg)
            d_negs.append(d_neg_i.mean())
        D_negs = torch.stack(d_negs)

        comp_losses.append(outfit_ranking_loss(D_pos.unsqueeze(0), D_negs.unsqueeze(0), aggregation=aggregation))
        d_pos_list.append(D_pos.item())
        d_neg_list.append(D_negs.min().item() if aggregation == "min" else D_negs.mean().item())

    complement_loss = torch.stack(comp_losses).mean() if comp_losses else torch.tensor(0.0, device=device)

    # ---- substitute sub-loss (ranking distillation over same-category neighbors) ----
    sub_losses = []
    for pair in substitute_pairs:
        if pair[0] is None:
            continue
        anchor_gidx, nbr_gidx, teacher_sims = pair
        anchor_cat = state.item_cat.get(state_item_id_for(state, anchor_gidx))
        cat_pair = state.onehot(anchor_cat).to(device)  # (1, C)
        n_nbr = len(nbr_gidx)

        x_anchor = x_all[gidx_to_pos[anchor_gidx]].unsqueeze(0)
        x_nbrs = x_all[[gidx_to_pos[int(g)] for g in nbr_gidx]]

        z_anchor = model.embed_from_feature(x_anchor, cat_pair, cat_pair, alpha_forward)  # (1, D)
        cat_pair_rep = cat_pair.expand(n_nbr, -1)
        z_nbrs = model.embed_from_feature(x_nbrs, cat_pair_rep, cat_pair_rep, alpha_forward)  # (K, D)

        teacher = torch.tensor(teacher_sims, device=device).unsqueeze(0)  # (1, K)
        sub_losses.append(ranking_distillation_loss(z_anchor, z_nbrs.unsqueeze(0), teacher, tau=TAU_DISTILL))
        representative_embeddings.append(z_anchor.squeeze(0))

    substitute_loss = torch.stack(sub_losses).mean() if sub_losses else torch.tensor(0.0, device=device)

    a = float(alpha_weight)
    total_loss = a * weight_sub * substitute_loss + (1 - a) * complement_loss
    if representative_embeddings and uniformity_weight > 0:
        total_loss = total_loss + uniformity_weight * uniformity_loss(torch.stack(representative_embeddings))

    d_pos_mean = float(np.mean(d_pos_list)) if d_pos_list else float("nan")
    d_neg_mean = float(np.mean(d_neg_list)) if d_neg_list else float("nan")
    # complement_loss/substitute_loss returned as raw tensors too (in addition to their
    # .item() values folded into total_loss's history) so callers that need to isolate
    # one term's gradient (e.g. the loss-balancing verification) can do so WITHOUT the
    # uniformity term's cross-branch coupling contaminating the measurement -- uniformity
    # is computed from the combined representative-embedding set from BOTH branches, so
    # backward()-ing through `total_loss` alone never isolates either term cleanly.
    return total_loss, d_pos_mean, d_neg_mean, complement_loss.item(), substitute_loss.item(), complement_loss, substitute_loss


def state_item_id_for(state, gidx):
    """Reverse lookup gidx -> item_id. Built lazily and cached on `state`."""
    if not hasattr(state, "_rev_idx"):
        state._rev_idx = {g: iid for iid, g in state.idx.items()}
    return state._rev_idx[gidx]


def build_substitute_pairs(state, samples):
    """One substitute pair per complement sample, anchored on that sample's
    positive item (design decision 3: reuse complement's anchors)."""
    pairs = []
    for s in samples:
        nbr_gidx, teacher_sims = state.substitute_pair_for(s["positive"])
        if nbr_gidx is None:
            pairs.append((None, None, None))
        else:
            pairs.append((state.idx[s["positive"]], nbr_gidx, teacher_sims))
    return pairs


def make_batch(state, outfits, batch_ids, split, device, alpha, gradient_needed=True):
    """One full step's worth of data + loss, factored out so calibration,
    smoke test, and the real training loop can all share it."""
    samples = [state.make_sample(outfits[i], split) for i in batch_ids]
    substitute_pairs = build_substitute_pairs(state, samples)
    vecs, gidx_to_pos = build_batch_vectors(state, samples, substitute_pairs, device)
    return samples, substitute_pairs, vecs, gidx_to_pos


def measure_initial_magnitudes(model, state, train_outfits, device, n_batches, batch_size, alpha=0.5):
    """Step 3 / design decision 6: calibrate weight_sub at alpha=0.5 ONLY --
    the one point where the loss envelope treats both terms symmetrically, so
    scale-matching and alpha-driven emphasis are cleanly separable."""
    model.eval()
    order = list(range(len(train_outfits)))
    state.rng.shuffle(order)
    comp_vals, sub_vals = [], []
    with torch.no_grad():
        for b in range(n_batches):
            batch_ids = order[b * batch_size:(b + 1) * batch_size]
            if len(batch_ids) < 2:
                continue
            samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
                state, train_outfits, batch_ids, "train", device, alpha)
            x_all = model.encode_feature(vecs)
            _, _, _, comp_loss, sub_loss, _, _ = compute_batch_loss(
                model, state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                alpha_forward=alpha, alpha_weight=alpha, weight_sub=1.0)
            comp_vals.append(comp_loss)
            sub_vals.append(sub_loss)
    return float(np.mean(comp_vals)), float(np.mean(sub_vals))


def grad_norm_check(model, state, train_outfits, device, batch_size, weight_sub, alpha):
    """Verification (not re-calibration, see design decision 6): gradient L2
    norm into the shared params (proj, attn_net, masks) from each RAW loss
    term in isolation (complement_loss and substitute_loss tensors directly,
    bypassing the uniformity term entirely -- uniformity couples both
    branches together via a combined representative-embedding set, so
    backward()-ing through the combined `total_loss` would never isolate
    either term cleanly)."""
    order = list(range(len(train_outfits)))
    state.rng.shuffle(order)
    batch_ids = order[:batch_size]
    samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
        state, train_outfits, batch_ids, "train", device, alpha)

    shared_params = list(model.proj.parameters()) + list(model.attn_net.parameters()) + [model.masks]

    def net_grad_norm(loss):
        model.zero_grad()
        loss.backward(retain_graph=False)
        total = 0.0
        for p in shared_params:
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    model.train()
    # Fresh encode_feature graph per isolated backward pass (each backward()+zero_grad()
    # consumes the previous graph). alpha_forward=alpha throughout, since we want the
    # conditioning input realistic for this alpha even while isolating each raw loss term.
    x_all_a = model.encode_feature(vecs)
    _, _, _, _, _, comp_loss_a, _ = compute_batch_loss(
        model, state, samples, substitute_pairs, x_all_a, gidx_to_pos, device,
        alpha_forward=alpha, alpha_weight=alpha, weight_sub=1.0)
    comp_gn = net_grad_norm(comp_loss_a)

    x_all_b = model.encode_feature(vecs)
    _, _, _, _, _, _, sub_loss_b = compute_batch_loss(
        model, state, samples, substitute_pairs, x_all_b, gidx_to_pos, device,
        alpha_forward=alpha, alpha_weight=alpha, weight_sub=1.0)
    sub_gn_unweighted = net_grad_norm(sub_loss_b)

    x_all_c = model.encode_feature(vecs)
    _, _, _, _, _, _, sub_loss_c = compute_batch_loss(
        model, state, samples, substitute_pairs, x_all_c, gidx_to_pos, device,
        alpha_forward=alpha, alpha_weight=alpha, weight_sub=1.0)
    sub_gn_weighted = net_grad_norm(weight_sub * sub_loss_c)

    model.zero_grad()
    return comp_gn, sub_gn_unweighted, sub_gn_weighted


def run_training(embeddings_npz, training_data_path, negative_candidates_path,
                  same_category_neighbors_npz, out_dir, device, max_epochs, batch_size, lr, patience,
                  weight_sub, alpha_mode="continuous", n_train_outfits=None, n_val_outfits=None,
                  log_every=20, decoupled_alpha_frac=0.0, uniformity_weight=UNIFORMITY_WEIGHT,
                  ckpt_suffix=""):
    """alpha_mode: 'continuous' samples alpha ~ Uniform(0,1) each step (the
    primary run). 'discrete' samples alpha from {0.0, 1.0} each step (the
    ablation). decoupled_alpha_frac > 0 runs that fraction of steps with the
    forward-pass alpha sampled independently from the loss-weighting alpha
    (design decision 5's conditioning-consistency safeguard) -- used by the
    smoke test, normally 0.0 for the real runs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"csa_net_controllable_{alpha_mode}{ckpt_suffix}_best.pt"

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
    val_rng = random.Random(1000)

    def sample_alpha(rng):
        if alpha_mode == "discrete":
            return rng.choice([0.0, 1.0])
        return rng.random()

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
            alpha_weight = sample_alpha(state.rng)
            if decoupled_alpha_frac > 0 and state.rng.random() < decoupled_alpha_frac:
                alpha_forward = sample_alpha(state.rng)
            else:
                alpha_forward = alpha_weight

            samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
                state, train_outfits, b_idx, "train", device, alpha_weight)

            optimizer.zero_grad()
            x_all = model.encode_feature(vecs)
            loss, d_pos, d_neg, comp_l, sub_l, _, _ = compute_batch_loss(
                model, state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                alpha_forward=alpha_forward, alpha_weight=alpha_weight, weight_sub=weight_sub,
                uniformity_weight=uniformity_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f} alpha={alpha_weight:.2f} "
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
                alpha = sample_alpha(val_rng)
                samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
                    val_state, val_outfits, b_idx, "val", device, alpha)
                x_all = model.encode_feature(vecs)
                loss, d_pos, d_neg, comp_l, sub_l, _, _ = compute_batch_loss(
                    model, val_state, samples, substitute_pairs, x_all, gidx_to_pos, device,
                    alpha_forward=alpha, alpha_weight=alpha, weight_sub=weight_sub,
                    uniformity_weight=uniformity_weight)
                val_losses.append(loss.item())
                val_d_pos.append(d_pos)
                val_d_neg.append(d_neg)
                val_comp.append(comp_l)
                val_sub.append(sub_l)
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(epoch_losses))
        print(f"== epoch {epoch} done ({alpha_mode}): train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_D_pos={np.mean(val_d_pos):.4f} val_D_neg={np.mean(val_d_neg):.4f} "
              f"val_comp={np.mean(val_comp):.4f} val_sub={np.mean(val_sub):.4f} ==", flush=True)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_D_pos": float(np.mean(val_d_pos)), "val_D_neg": float(np.mean(val_d_neg)),
            "val_complement_loss": float(np.mean(val_comp)), "val_substitute_loss": float(np.mean(val_sub)),
        })
        with open(out_dir / f"training_curves_{alpha_mode}.json", "w") as f:
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

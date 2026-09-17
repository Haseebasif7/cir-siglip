"""
Phase 17, steps 1-4: retrofit phase 16d's dedicated-capacity architecture
onto phase 12c's exact substitute/complement mechanism. Data, both loss
formulations (complement = MNRL/InfoNCE on real outfit co-occurrence,
substitute = KL ranking-distillation toward each anchor's actual top-50
raw-SigLIP neighbors), hyperparameters, and early-stopping procedure are
copied UNCHANGED from phase 12c's `03_train_ranking_distillation.py` -- the
only change is the model class (`DedicatedCapacityHead` instead of
`ControllableProjectionHead`) and, following directly from that, the
gradient-norm check now measures gradient flowing into `model.shared`
(the new minimal shared layer) instead of `model.net` (phase 12c's full
shared trunk) -- the correct adaptation of the same check to this
architecture, not a relaxation of it.

The nearest-neighbor teacher lookup (`nn_lookup.npz`) is reused directly from
phase 12c's own `data/nn_lookup.npz`, unchanged -- it depends only on the raw,
frozen SigLIP embeddings (unchanged since phase 12c), not on anything this
phase trains, so recomputing it would be identical work for no reason. Same
reuse pattern as `cir_eval.py` pointing directly at phase 12's benchmark file.

Loss balancing is calibrated FRESH (not assumed to match phase 12c's 64.47x
weight) per the brief's explicit instruction -- this is a different
architecture, so the same loss pair can produce a different gradient
relationship into the (now much smaller) shared layer.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DedicatedCapacityHead, mean_pairwise_cosine, mnrl_loss

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12C_DIR = BASE_DIR.parent.parent / "week4" / "phase12c_ranking_distillation"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = PHASE9_DIR / "data" / "positive_edges.json"
NN_LOOKUP_NPZ = PHASE12C_DIR / "data" / "nn_lookup.npz"  # reused unchanged, see docstring
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOSS_BALANCING_MD = BASE_DIR / "loss_balancing_check.md"

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU_COMPLEMENT = 0.07
TAU_DISTILL = 0.07
R_NEG = 8
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-4
N_CALIBRATION_BATCHES = 20
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Phase 12c's own calibration/verification numbers, for direct comparison.
PHASE12C_INITIAL_COMP = 4.8981
PHASE12C_INITIAL_SUB = 0.0760
PHASE12C_WEIGHT_SUB = 64.4731
PHASE12C_RATIO_AFTER = 0.1


def load_data():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    nn = np.load(NN_LOOKUP_NPZ, allow_pickle=True)
    nn_item_ids = [str(a) for a in nn["item_ids"]]
    assert nn_item_ids == item_ids, "NN lookup ordering must match embedding ordering"
    nn_indices = nn["indices"]
    nn_sims = nn["sims"].astype(np.float32)

    with open(POSITIVE_EDGES_JSON) as f:
        edge_records = json.load(f)
    train_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "train"]
    val_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "val"]

    all_positive_targets = {}
    for src, tgt in train_edges + val_edges:
        all_positive_targets.setdefault(src, set()).add(tgt)

    return embeddings, nn_indices, nn_sims, idx, item_ids, train_edges, val_edges, all_positive_targets


def sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_item, k):
    negs = []
    exclude = positive_set | {anchor}
    tries, max_tries = 0, k * 20
    while len(negs) < k and tries < max_tries:
        cand = idx_to_item[rng.randrange(n_items)]
        if cand not in exclude and cand not in negs:
            negs.append(cand)
        tries += 1
    while len(negs) < k:
        cand = idx_to_item[rng.randrange(n_items)]
        negs.append(cand)
    return negs


def build_negatives_for_edges(edges, all_positive_targets, item_ids, rng, k):
    n_items = len(item_ids)
    out = []
    for anchor, _ in edges:
        positive_set = all_positive_targets.get(anchor, set())
        out.append(sample_random_negatives(rng, anchor, positive_set, n_items, item_ids, k))
    return out


def build_inbatch_mask(batch_edges, all_positive_targets):
    B = len(batch_edges)
    mask = np.zeros((B, B), dtype=bool)
    for i, (anchor_i, _) in enumerate(batch_edges):
        positives_i = all_positive_targets.get(anchor_i, set())
        for j, (_, positive_j) in enumerate(batch_edges):
            if i != j and positive_j in positives_i:
                mask[i, j] = True
    return mask


def complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets):
    anchors = [e[0] for e in batch_edges]
    positives = [e[1] for e in batch_edges]
    a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
    p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
    extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
    extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

    z_a = model(a_emb, alpha=0.0)
    z_p = model(p_emb, alpha=0.0)
    B, K, D = extra_emb.shape
    z_extra = model(extra_emb.reshape(B * K, D), alpha=0.0).reshape(B, K, -1)
    mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
    return mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU_COMPLEMENT)


def substitute_ranking_loss(model, embeddings, idx, batch_edges, nn_indices, nn_sims):
    anchors = [e[0] for e in batch_edges]
    anchor_gidx = np.array([idx[a] for a in anchors])
    neighbor_gidx = nn_indices[anchor_gidx]
    B, K = neighbor_gidx.shape

    teacher_sims = torch.tensor(nn_sims[anchor_gidx], device=DEVICE)
    teacher_dist = F.softmax(teacher_sims / TAU_DISTILL, dim=-1)

    anchor_emb = torch.tensor(embeddings[anchor_gidx], device=DEVICE)
    neighbor_emb = torch.tensor(embeddings[neighbor_gidx.reshape(-1)], device=DEVICE)

    z_anchor = model(anchor_emb, alpha=1.0)
    z_neighbors = model(neighbor_emb, alpha=1.0).reshape(B, K, -1)

    student_sims = torch.einsum("bd,bkd->bk", z_anchor, z_neighbors)
    student_log_probs = F.log_softmax(student_sims / TAU_DISTILL, dim=-1)

    return F.kl_div(student_log_probs, teacher_dist, reduction="batchmean")


def measure_initial_magnitudes(model, embeddings, nn_indices, nn_sims, idx, edges, all_positive_targets,
                                item_ids, rng, n_batches):
    model.eval()
    order = list(range(len(edges)))
    rng.shuffle(order)
    comp_vals, sub_vals = [], []
    with torch.no_grad():
        for b in range(n_batches):
            batch_idx = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            batch_edges = [edges[i] for i in batch_idx]
            neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)
            comp_loss = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
            sub_loss = substitute_ranking_loss(model, embeddings, idx, batch_edges, nn_indices, nn_sims)
            comp_vals.append(comp_loss.item())
            sub_vals.append(sub_loss.item())
    return float(np.mean(comp_vals)), float(np.mean(sub_vals))


def grad_norm_check(model, embeddings, nn_indices, nn_sims, idx, edges, all_positive_targets, item_ids, rng, weight_sub):
    order = list(range(len(edges)))
    rng.shuffle(order)
    batch_edges = [edges[i] for i in order[:BATCH_SIZE]]
    neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

    def shared_grad_norm(loss):
        model.zero_grad()
        loss.backward()
        total = 0.0
        for p in model.shared.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    model.train()
    comp_loss = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    comp_grad_norm = shared_grad_norm(comp_loss)

    sub_loss = substitute_ranking_loss(model, embeddings, idx, batch_edges, nn_indices, nn_sims)
    sub_grad_norm_unweighted = shared_grad_norm(sub_loss)

    sub_loss2 = substitute_ranking_loss(model, embeddings, idx, batch_edges, nn_indices, nn_sims)
    sub_grad_norm_weighted = shared_grad_norm(weight_sub * sub_loss2)

    model.zero_grad()
    return comp_grad_norm, sub_grad_norm_unweighted, sub_grad_norm_weighted


def run_epoch_train(model, optimizer, embeddings, nn_indices, nn_sims, idx, edges, all_positive_targets,
                     item_ids, rng, weight_sub):
    model.train()
    order = list(range(len(edges)))
    rng.shuffle(order)
    total_comp, total_sub, n_batches = 0.0, 0.0, 0

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue
        batch_edges = [edges[i] for i in batch_idx]
        neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

        complement_loss = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
        substitute_loss = substitute_ranking_loss(model, embeddings, idx, batch_edges, nn_indices, nn_sims)

        loss = complement_loss + weight_sub * substitute_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_comp += complement_loss.item()
        total_sub += substitute_loss.item()
        n_batches += 1
        if n_batches % 2000 == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, "
                  f"complement={total_comp/n_batches:.4f} substitute={total_sub/n_batches:.4f}", flush=True)

    return total_comp / max(n_batches, 1), total_sub / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, nn_indices, nn_sims, idx, val_edges, frozen_val_negs, all_positive_targets):
    model.eval()
    total_comp, total_sub, n_batches = 0.0, 0.0, 0
    for start in range(0, len(val_edges), BATCH_SIZE):
        batch_edges = val_edges[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_val_negs[start:start + BATCH_SIZE]
        complement_loss = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
        substitute_loss = substitute_ranking_loss(model, embeddings, idx, batch_edges, nn_indices, nn_sims)
        total_comp += complement_loss.item()
        total_sub += substitute_loss.item()
        n_batches += 1
    return total_comp / max(n_batches, 1), total_sub / max(n_batches, 1)


def main():
    embeddings, nn_indices, nn_sims, idx, item_ids, train_edges, val_edges, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, "
          f"{len(val_edges)} val edges, NN lookup K={nn_indices.shape[1]} (reused from phase 12c). "
          f"Device: {DEVICE}", flush=True)

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)
    calib_rng = random.Random(SEED + 2000)
    grad_rng = random.Random(SEED + 3000)

    model = DedicatedCapacityHead().to(DEVICE)

    initial_comp, initial_sub = measure_initial_magnitudes(
        model, embeddings, nn_indices, nn_sims, idx, train_edges, all_positive_targets, item_ids,
        calib_rng, N_CALIBRATION_BATCHES)
    weight_sub = initial_comp / initial_sub
    print(f"Calibration ({N_CALIBRATION_BATCHES} batches): initial_comp={initial_comp:.4f}, "
          f"initial_sub={initial_sub:.4f}, weight_sub={weight_sub:.4f}", flush=True)

    comp_gn, sub_gn_unweighted, sub_gn_weighted = grad_norm_check(
        model, embeddings, nn_indices, nn_sims, idx, train_edges, all_positive_targets, item_ids,
        grad_rng, weight_sub)
    ratio_before = comp_gn / sub_gn_unweighted if sub_gn_unweighted > 0 else float("inf")
    ratio_after = comp_gn / sub_gn_weighted if sub_gn_weighted > 0 else float("inf")
    print(f"Grad norm into shared layer -- complement: {comp_gn:.6f}, "
          f"substitute (unweighted): {sub_gn_unweighted:.6f} (ratio {ratio_before:.1f}x), "
          f"substitute (weighted x{weight_sub:.2f}): {sub_gn_weighted:.6f} (ratio {ratio_after:.1f}x)", flush=True)

    lines = [
        "# Phase 17: Loss Balancing Check (Fresh Calibration for Dedicated-Capacity Architecture)",
        "",
        f"Measured fresh on the freshly-initialized (untrained) `DedicatedCapacityHead`, seed={SEED} -- "
        "NOT assumed to match phase 12c's weight (64.4731x), per the brief's explicit instruction: "
        "this is the same loss pair on a structurally different architecture, so the gradient "
        "relationship into the shared parameters can differ even if the loss VALUES don't.",
        "",
        "## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)",
        "",
        f"- Complement loss (MNRL/InfoNCE): {initial_comp:.4f} (phase 12c: {PHASE12C_INITIAL_COMP:.4f})",
        f"- Substitute loss (KL ranking distillation, K={nn_indices.shape[1]}, tau={TAU_DISTILL}): "
        f"{initial_sub:.4f} (phase 12c: {PHASE12C_INITIAL_SUB:.4f})",
        f"- Ratio: {initial_comp/initial_sub:.2f}x (phase 12c's ratio on `ControllableProjectionHead`: "
        f"{PHASE12C_INITIAL_COMP/PHASE12C_INITIAL_SUB:.2f}x). Loss VALUES are architecture-independent "
        "at init (both losses only depend on the random projection's outputs, and the two "
        "architectures' output distributions at random init are not expected to differ enough to "
        "change this ratio much) -- so a similar ratio here is expected and not itself informative; "
        "the gradient-norm check below is the real test.",
        "",
        f"- **Chosen weight_sub = initial_comp / initial_sub = {weight_sub:.4f}**.",
        "",
        "## Gradient-norm verification",
        "",
        "Gradient L2-norm into the SHARED layer's parameters only (`model.shared`, the 768->256 "
        "dimensionality-reduction step -- NOT `model.substitute_head`/`model.complement_head`, "
        "which are per-mode and not shared), each loss term isolated, one fixed batch of 128 "
        "anchor edges. This is the adapted equivalent of phase 12c's check on `model.net` -- the "
        "correct comparison point is the shared parameters specifically, and phase 17's shared "
        "layer is much smaller (768->256 only) than phase 12c's full shared trunk (768->256->128), "
        "so a different verification outcome here is possible even with an identical loss pair:",
        "",
        "| Loss term | Grad norm into shared layer | Ratio to complement |",
        "|---|---|---|",
        f"| Complement (MNRL) | {comp_gn:.6f} | 1.0x (reference) |",
        f"| Substitute, UNWEIGHTED | {sub_gn_unweighted:.6f} | {ratio_before:.1f}x smaller |",
        f"| Substitute, weighted x{weight_sub:.2f} | {sub_gn_weighted:.6f} | {ratio_after:.1f}x |",
        "",
        f"(Phase 12c's own check, for reference, on `model.net`: weighted ratio landed at "
        f"{PHASE12C_RATIO_AFTER:.1f}x, weight_sub={PHASE12C_WEIGHT_SUB:.2f}.)",
        "",
    ]
    if 0.1 <= ratio_after <= 10:
        lines.append(f"**Verification PASSED**: after weighting, the substitute loss's gradient "
                      f"into the shared layer is within a reasonable order of magnitude of the "
                      f"complement loss's ({ratio_after:.2f}x). Proceeding to full training with "
                      "this fixed weight_sub.")
    else:
        lines.append(f"**Verification DID NOT FULLY PASS**: even after weighting, the two "
                      f"gradients remain {ratio_after:.1f}x apart. Proceeding to train with this "
                      "weight regardless and reporting this discrepancy plainly -- see "
                      "phase17_notes.md for how this affects the final verdict.")
    lines.append("")
    LOSS_BALANCING_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {LOSS_BALANCING_MD}", flush=True)

    frozen_val_negs = build_negatives_for_edges(val_edges, all_positive_targets, item_ids, val_rng, R_NEG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_comp = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_comp, train_sub = run_epoch_train(model, optimizer, embeddings, nn_indices, nn_sims, idx,
                                                  train_edges, all_positive_targets, item_ids, rng, weight_sub)
        val_comp, val_sub = run_epoch_val(model, embeddings, nn_indices, nn_sims, idx, val_edges,
                                           frozen_val_negs, all_positive_targets)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_comp = mean_pairwise_cosine(model(sample_emb, alpha=0.0))
            collapse_sub = mean_pairwise_cosine(model(sample_emb, alpha=1.0))

        curves.append({
            "epoch": epoch, "train_complement_loss": train_comp, "train_substitute_loss": train_sub,
            "val_complement_loss": val_comp, "val_substitute_loss": val_sub, "weight_sub": weight_sub,
            "mean_pairwise_cosine_complement": collapse_comp, "mean_pairwise_cosine_substitute": collapse_sub,
        })
        print(f"  epoch {epoch}: train_comp={train_comp:.4f} train_sub={train_sub:.4f} "
              f"val_comp={val_comp:.4f} val_sub={val_sub:.4f} "
              f"collapse_comp={collapse_comp:.4f} collapse_sub={collapse_sub:.4f}", flush=True)

        if val_comp < best_val_comp - MIN_DELTA:
            best_val_comp = val_comp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best_val_comp={best_val_comp:.4f})", flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "dedicated_capacity_substitute_complement.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'dedicated_capacity_substitute_complement.pt'} "
          f"(best_val_comp={best_val_comp:.4f})", flush=True)

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}", flush=True)


if __name__ == "__main__":
    main()

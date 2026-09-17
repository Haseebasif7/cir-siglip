"""
Phase 12b, steps 1-3: retrain the controllable mode-embedding mechanism with
phase 12's diagnosed fix applied.

Two changes from phase 12's `03_train_controllable_modes.py`, and ONLY these
two (everything else -- architecture, complement loss, positive edges,
hyperparameters, early stopping -- is unchanged, so any change in outcome can
be attributed specifically to these fixes):

1. **Substitute loss reformulated** (step 1): instead of matching pairwise
   similarity structure within a single training batch (diagnosed as weak and
   indirect), substitute mode now regresses each item's own embedding
   directly toward a FIXED, global per-item target -- a PCA-128 projection of
   that item's raw SigLIP embedding, computed once in
   `01_compute_raw_targets.py` (see that script's docstring for why a fixed
   projection was needed to make "match the item's own raw embedding
   directly" dimensionally well-defined, since z_sub is 128-d and raw SigLIP
   is 768-d). Loss: `1 - cosine(z_sub_i, pca_target_i)`, averaged over the
   batch's anchor items.

2. **Loss balancing measured and verified, not assumed** (step 2): before any
   optimizer step, this script measures both losses' typical starting
   magnitude under the NEW substitute-loss formulation (phase 12's old ~400x
   ratio doesn't necessarily still hold once the substitute loss itself
   changed), computes a single fixed scalar weight so both losses contribute
   comparably at the start of training, and then explicitly checks --
   BEFORE committing to the full run -- that the gradient norm flowing into
   the shared base projection from each loss (isolated, with the other
   zeroed) is now within a reasonable order of magnitude of the other, not
   still off by hundreds of times. All of this is written to
   `../loss_balancing_check.md` as it happens, not reconstructed after.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ControllableProjectionHead, mean_pairwise_cosine, mnrl_loss

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = PHASE9_DIR / "data" / "positive_edges.json"
PCA_TARGETS_NPZ = BASE_DIR / "data" / "raw_pca128_targets.npz"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOSS_BALANCING_MD = BASE_DIR / "loss_balancing_check.md"

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU = 0.07
R_NEG = 8  # unchanged from phase 12 Model A / phase 12
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-4
N_CALIBRATION_BATCHES = 20  # for measuring initial loss magnitudes
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_data():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    pca = np.load(PCA_TARGETS_NPZ, allow_pickle=True)
    pca_item_ids = [str(a) for a in pca["item_ids"]]
    assert pca_item_ids == item_ids, "PCA target ordering must match embedding ordering"
    pca_targets = pca["targets"].astype(np.float32)  # already L2-normalized, (N, 128)

    with open(POSITIVE_EDGES_JSON) as f:
        edge_records = json.load(f)
    train_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "train"]
    val_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "val"]

    all_positive_targets = {}
    for src, tgt in train_edges + val_edges:
        all_positive_targets.setdefault(src, set()).add(tgt)

    return embeddings, pca_targets, idx, item_ids, train_edges, val_edges, all_positive_targets


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
    return mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU), a_emb


def substitute_loss_for_batch(model, embeddings, pca_targets, idx, batch_edges):
    """Step 1's fix: direct per-item regression toward a fixed PCA-128 target,
    not batch-local pairwise-similarity matching."""
    anchors = [e[0] for e in batch_edges]
    a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
    target = torch.tensor(pca_targets[[idx[a] for a in anchors]], device=DEVICE)
    z_sub = model(a_emb, alpha=1.0)
    cos = (z_sub * target).sum(dim=-1)
    return (1 - cos).mean()


def measure_initial_magnitudes(model, embeddings, pca_targets, idx, edges, all_positive_targets,
                                item_ids, rng, n_batches):
    """Step 2, part 1: typical starting magnitude of each loss, averaged over
    a handful of batches, on the freshly-initialized (untrained) model."""
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
            comp_loss, _ = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
            sub_loss = substitute_loss_for_batch(model, embeddings, pca_targets, idx, batch_edges)
            comp_vals.append(comp_loss.item())
            sub_vals.append(sub_loss.item())
    return float(np.mean(comp_vals)), float(np.mean(sub_vals))


def grad_norm_check(model, embeddings, pca_targets, idx, edges, all_positive_targets, item_ids, rng, weight_sub):
    """Step 2, part 2: gradient norm into the shared base projection (`net`)
    from each loss term separately (other term zeroed), on one fixed batch,
    both BEFORE and AFTER applying weight_sub -- this is the explicit
    verification the brief requires, not an assumption."""
    order = list(range(len(edges)))
    rng.shuffle(order)
    batch_edges = [edges[i] for i in order[:BATCH_SIZE]]
    neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

    def net_grad_norm(loss):
        model.zero_grad()
        loss.backward()
        total = 0.0
        for p in model.net.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    model.train()
    comp_loss, _ = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    comp_grad_norm = net_grad_norm(comp_loss)

    sub_loss = substitute_loss_for_batch(model, embeddings, pca_targets, idx, batch_edges)
    sub_grad_norm_unweighted = net_grad_norm(sub_loss)

    sub_loss2 = substitute_loss_for_batch(model, embeddings, pca_targets, idx, batch_edges)
    sub_grad_norm_weighted = net_grad_norm(weight_sub * sub_loss2)

    model.zero_grad()
    return comp_grad_norm, sub_grad_norm_unweighted, sub_grad_norm_weighted


def run_epoch_train(model, optimizer, embeddings, pca_targets, idx, edges, all_positive_targets,
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

        complement_loss, _ = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
        substitute_loss = substitute_loss_for_batch(model, embeddings, pca_targets, idx, batch_edges)

        loss = complement_loss + weight_sub * substitute_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_comp += complement_loss.item()
        total_sub += substitute_loss.item()
        n_batches += 1
        if n_batches % 2000 == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, "
                  f"complement={total_comp/n_batches:.4f} substitute={total_sub/n_batches:.4f}")

    return total_comp / max(n_batches, 1), total_sub / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, pca_targets, idx, val_edges, frozen_val_negs, all_positive_targets, weight_sub):
    model.eval()
    total_comp, total_sub, n_batches = 0.0, 0.0, 0
    for start in range(0, len(val_edges), BATCH_SIZE):
        batch_edges = val_edges[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_val_negs[start:start + BATCH_SIZE]
        complement_loss, _ = complement_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
        substitute_loss = substitute_loss_for_batch(model, embeddings, pca_targets, idx, batch_edges)
        total_comp += complement_loss.item()
        total_sub += substitute_loss.item()
        n_batches += 1
    return total_comp / max(n_batches, 1), total_sub / max(n_batches, 1)


def main():
    embeddings, pca_targets, idx, item_ids, train_edges, val_edges, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, "
          f"{len(val_edges)} val edges. Device: {DEVICE}")

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)
    calib_rng = random.Random(SEED + 2000)
    grad_rng = random.Random(SEED + 3000)

    model = ControllableProjectionHead().to(DEVICE)

    # Step 2, part 1: measure initial magnitudes on the untrained model.
    initial_comp, initial_sub = measure_initial_magnitudes(
        model, embeddings, pca_targets, idx, train_edges, all_positive_targets, item_ids,
        calib_rng, N_CALIBRATION_BATCHES)
    weight_sub = initial_comp / initial_sub
    print(f"Calibration ({N_CALIBRATION_BATCHES} batches): initial_comp={initial_comp:.4f}, "
          f"initial_sub={initial_sub:.4f}, weight_sub={weight_sub:.4f}")

    # Step 2, part 2: explicit gradient-norm verification, before vs after weighting.
    comp_gn, sub_gn_unweighted, sub_gn_weighted = grad_norm_check(
        model, embeddings, pca_targets, idx, train_edges, all_positive_targets, item_ids,
        grad_rng, weight_sub)
    ratio_before = comp_gn / sub_gn_unweighted if sub_gn_unweighted > 0 else float("inf")
    ratio_after = comp_gn / sub_gn_weighted if sub_gn_weighted > 0 else float("inf")
    print(f"Grad norm into shared net -- complement: {comp_gn:.6f}, "
          f"substitute (unweighted): {sub_gn_unweighted:.6f} (ratio {ratio_before:.1f}x), "
          f"substitute (weighted x{weight_sub:.2f}): {sub_gn_weighted:.6f} (ratio {ratio_after:.1f}x)")

    lines = [
        "# Phase 12b, Step 2: Loss Balancing Check",
        "",
        f"Measured on the freshly-initialized (untrained) `ControllableProjectionHead`, "
        f"seed={SEED}.",
        "",
        "## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)",
        "",
        f"- Complement loss (MNRL/InfoNCE): {initial_comp:.4f}",
        f"- Substitute loss (1 - cosine to fixed PCA-128 target, phase 12b's new formulation): {initial_sub:.4f}",
        f"- Ratio: {initial_comp/initial_sub:.1f}x "
        f"(phase 12's old batch-local pairwise-MSE substitute loss started at a very different "
        "scale -- ~0.015 vs complement's ~4.4, a ~290x ratio at that point in training; this "
        "new formulation's starting ratio is not assumed to be the same and is remeasured here "
        "directly, per the brief's explicit instruction).",
        "",
        f"- **Chosen weight_sub = initial_comp / initial_sub = {weight_sub:.4f}** (single fixed "
        "scalar, applied to the substitute loss for the entire training run -- normalizes so "
        "both loss terms contribute comparably to the shared network's gradient at the start "
        "of training, the 'normalize by each loss's own starting magnitude' approach the brief "
        "suggested).",
        "",
        "## Gradient-norm verification (the actual check, not just the loss-value ratio)",
        "",
        "Gradient L2-norm flowing into the shared base projection's parameters (`model.net`), "
        "computed from each loss term SEPARATELY (the other term excluded from that backward "
        "pass), on one fixed batch of 128 anchor edges:",
        "",
        "| Loss term | Grad norm into shared net | Ratio to complement |",
        "|---|---|---|",
        f"| Complement (MNRL) | {comp_gn:.6f} | 1.0x (reference) |",
        f"| Substitute, UNWEIGHTED | {sub_gn_unweighted:.6f} | {ratio_before:.1f}x smaller |",
        f"| Substitute, weighted x{weight_sub:.2f} | {sub_gn_weighted:.6f} | {ratio_after:.1f}x |",
        "",
    ]
    if 0.1 <= ratio_after <= 10:
        lines.append(f"**Verification PASSED**: after weighting, the substitute loss's gradient "
                      f"into the shared net is within a reasonable order of magnitude of the "
                      f"complement loss's ({ratio_after:.2f}x, not hundreds of times off). "
                      "Proceeding to full training with this fixed weight_sub.")
    else:
        lines.append(f"**Verification DID NOT FULLY PASS**: even after weighting by the initial "
                      f"loss-magnitude ratio, the two gradients remain {ratio_after:.1f}x apart "
                      "-- loss-VALUE magnitude and gradient-NORM magnitude are not the same "
                      "thing (a loss can be numerically small while still producing a large "
                      "gradient, or vice versa, depending on the function's curvature/scale). "
                      "Proceeding to train with this weight regardless (it is still a large "
                      "improvement over the unweighted ratio) and reporting this discrepancy "
                      "plainly rather than silently treating the fix as fully verified -- see "
                      "phase12b_notes.md for how this affects the final verdict.")
    lines.append("")
    LOSS_BALANCING_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {LOSS_BALANCING_MD}")

    # Step 3: full joint training, same procedure as phase 12.
    frozen_val_negs = build_negatives_for_edges(val_edges, all_positive_targets, item_ids, val_rng, R_NEG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_comp = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_comp, train_sub = run_epoch_train(model, optimizer, embeddings, pca_targets, idx,
                                                  train_edges, all_positive_targets, item_ids, rng, weight_sub)
        val_comp, val_sub = run_epoch_val(model, embeddings, pca_targets, idx, val_edges,
                                           frozen_val_negs, all_positive_targets, weight_sub)

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
              f"collapse_comp={collapse_comp:.4f} collapse_sub={collapse_sub:.4f}")

        if val_comp < best_val_comp - MIN_DELTA:
            best_val_comp = val_comp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best_val_comp={best_val_comp:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "controllable_modes_fixed.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'controllable_modes_fixed.pt'} (best_val_comp={best_val_comp:.4f})")

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

"""Phase 18, steps 4-5: full training run. Fresh loss-balancing calibration
for all four corners (not assumed from any prior phase's weights -- new
architecture, new corner combinations), then the full train/val loop with
early stopping.

`comp_rel` (complement-relevance -- MNRL on the full also_buy set) is
chosen as the fixed reference (weight=1.0), not calibrated. This is the
natural double-reference point: it's the exact signal phase 16's original
"relevance" mode used (the protected, non-reweighted objective in the
relevance/tail-exposure thread), AND it's the same MNRL-loss-family
objective phase 12/12c/17's "complement" mode was (the protected,
non-reweighted objective in the substitute/complement thread). The other
three corners (`sub_rel`, `comp_tail`, `sub_tail`) are each calibrated
against it independently: `weight_c = initial_comp_rel / initial_c`.

Early stopping tracks `val_comp_rel` alone, the same single-metric
convention every prior phase in this project used (phase 12c/17: val_comp;
phase 16/16d: val_rel) -- `comp_rel` is this phase's analog of both.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FourHeadDedicatedCapacity, mean_pairwise_cosine
from train_core import (
    BATCH_SIZE, CORNERS, DEVICE, LR, MAX_EPOCHS, MIN_DELTA, N_CALIBRATION_BATCHES, PATIENCE, R_NEG, SEED,
    WEIGHT_DECAY, build_negatives_for_edges, grad_norm_check, load_data, measure_initial_magnitudes,
    run_epoch_train, run_epoch_val,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOSS_BALANCING_MD = BASE_DIR / "loss_balancing_check.md"

REFERENCE_CORNER = "comp_rel"


def main():
    (embeddings, idx, item_ids, rel_train, rel_val, tail_train, tail_val, all_positive_targets,
     sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims) = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(rel_train)} relevance-axis train / {len(rel_val)} val, "
          f"{len(tail_train)} tail-axis train / {len(tail_val)} val. Device: {DEVICE}")

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)
    calib_rng = random.Random(SEED + 2000)
    grad_rng = random.Random(SEED + 3000)

    model = FourHeadDedicatedCapacity().to(DEVICE)

    initial = measure_initial_magnitudes(model, embeddings, idx, rel_train, tail_train, all_positive_targets,
                                          item_ids, sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims,
                                          calib_rng, N_CALIBRATION_BATCHES)
    weights = {c: (1.0 if c == REFERENCE_CORNER else initial[REFERENCE_CORNER] / initial[c]) for c in CORNERS}
    print(f"Calibration ({N_CALIBRATION_BATCHES} batches): " +
          ", ".join(f"initial_{c}={initial[c]:.4f}" for c in CORNERS))
    print("Weights: " + ", ".join(f"weight_{c}={weights[c]:.4f}" for c in CORNERS))

    grad_results = grad_norm_check(model, embeddings, idx, rel_train, tail_train, all_positive_targets, item_ids,
                                    sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims, grad_rng, weights)
    ref_gn = grad_results[f"{REFERENCE_CORNER}_unweighted"]

    lines = [
        "# Phase 18, Steps 4-5: Loss Balancing Check (Four Corners)",
        "",
        f"Measured fresh on the freshly-initialized (untrained) `FourHeadDedicatedCapacity`, seed={SEED} -- "
        "not assumed from any prior phase's weights, per this project's standing rule.",
        "",
        f"`{REFERENCE_CORNER}` (complement-relevance, MNRL on the full also_buy set) is the fixed reference "
        "(weight=1.0) -- the double-protected objective, matching both phase 16/16d's 'relevance' "
        "convention and phase 12c/17's 'complement' convention.",
        "",
        "## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)",
        "",
        "| Corner | Initial loss | Weight |",
        "|---|---|---|",
    ] + [f"| {c} | {initial[c]:.4f} | {weights[c]:.4f} |" for c in CORNERS] + [
        "",
        "## Gradient-norm verification (shared layer only, each corner isolated)",
        "",
        "Gradient L2-norm into `model.shared.parameters()` -- the only shared submodule -- one fixed "
        f"batch of {BATCH_SIZE} anchor edges per axis population:",
        "",
        "| Corner | Grad norm, UNWEIGHTED | Grad norm, WEIGHTED | Ratio to reference (weighted) |",
        "|---|---|---|---|",
    ]
    all_pass = True
    for c in CORNERS:
        if c == REFERENCE_CORNER:
            lines.append(f"| {c} | {ref_gn:.6f} | {ref_gn:.6f} | 1.0x (reference) |")
            continue
        unweighted = grad_results[f"{c}_unweighted"]
        weighted = grad_results[f"{c}_weighted"]
        ratio = ref_gn / weighted if weighted > 0 else float("inf")
        passed = 0.1 <= ratio <= 10
        all_pass = all_pass and passed
        lines.append(f"| {c} | {unweighted:.6f} | {weighted:.6f} | {ratio:.2f}x |")
    lines.append("")
    if all_pass:
        lines.append(f"**Verification PASSED for all 3 non-reference corners**: after weighting, every "
                      f"corner's gradient into the shared layer is within the [0.1, 10] band relative to "
                      f"`{REFERENCE_CORNER}`. Proceeding to full training with these fixed weights.")
    else:
        lines.append("**Verification DID NOT FULLY PASS for at least one corner** -- proceeding to train "
                      "with these weights regardless and reporting the discrepancy plainly in phase18_notes.md.")
    lines.append("")
    LOSS_BALANCING_MD.write_text("\n".join(lines))
    print(f"Saved {LOSS_BALANCING_MD}")
    print(f"All gradient-norm checks passed: {all_pass}")

    frozen_rel_val_negs = build_negatives_for_edges(rel_val, all_positive_targets, item_ids, val_rng, R_NEG)
    frozen_tail_val_negs = build_negatives_for_edges(tail_val, all_positive_targets, item_ids, val_rng, R_NEG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_ref = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_losses = run_epoch_train(model, optimizer, embeddings, idx, rel_train, tail_train,
                                        all_positive_targets, item_ids, sr_nn_indices, sr_nn_sims,
                                        st_nn_indices, st_nn_sims, rng, weights)
        val_losses = run_epoch_val(model, embeddings, idx, rel_val, tail_val, frozen_rel_val_negs,
                                    frozen_tail_val_negs, all_positive_targets, sr_nn_indices, sr_nn_sims,
                                    st_nn_indices, st_nn_sims)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse = {c: mean_pairwise_cosine(model.single_head_output(sample_emb, c)) for c in CORNERS}

        record = {"epoch": epoch}
        record.update({f"train_{c}_loss": train_losses[c] for c in CORNERS})
        record.update({f"val_{c}_loss": val_losses[c] for c in CORNERS})
        record.update({f"mean_pairwise_cosine_{c}": collapse[c] for c in CORNERS})
        record["weights"] = weights
        curves.append(record)

        train_summary = " ".join(f"{c}={train_losses[c]:.4f}" for c in CORNERS)
        val_summary = " ".join(f"{c}={val_losses[c]:.4f}" for c in CORNERS)
        print(f"  epoch {epoch}: train[{train_summary}] val[{val_summary}]")

        val_ref = val_losses[REFERENCE_CORNER]
        if val_ref < best_val_ref - MIN_DELTA:
            best_val_ref = val_ref
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best_val_{REFERENCE_CORNER}={best_val_ref:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "unified_two_axis.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'unified_two_axis.pt'} (best_val_{REFERENCE_CORNER}={best_val_ref:.4f})")

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

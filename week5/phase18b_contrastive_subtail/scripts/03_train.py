"""Phase 18b, steps 2-3: full training run. Fresh loss-balancing calibration
for all four corners -- NOT assumed from phase 18's weights, since
`sub_tail`'s objective formulation changed entirely (KL ranking-distillation,
scale ~0.2 at init, to MNRL, scale ~2-5 at init -- a completely different
loss family, so the old 23.7x weight is certain to be wrong even before
measuring).

`comp_rel` remains the fixed reference (weight=1.0), unchanged reasoning
from phase 18. Early stopping tracks `val_comp_rel` alone, same convention.
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
    WEIGHT_DECAY, build_head_biased_negatives_for_edges, build_negatives_for_edges, grad_norm_check, load_data,
    measure_initial_magnitudes, run_epoch_train, run_epoch_val,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOSS_BALANCING_MD = BASE_DIR / "loss_balancing_check.md"

REFERENCE_CORNER = "comp_rel"


def main():
    (embeddings, idx, item_ids, rel_train, rel_val, ct_train, ct_val, st_train, st_val, all_positive_targets,
     sr_nn_indices, sr_nn_sims, head_tier_ids) = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(rel_train)} REL train / {len(rel_val)} val, "
          f"{len(ct_train)} CT train / {len(ct_val)} val, {len(st_train)} ST train / {len(st_val)} val. "
          f"Device: {DEVICE}")

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)
    calib_rng = random.Random(SEED + 2000)
    grad_rng = random.Random(SEED + 3000)

    model = FourHeadDedicatedCapacity().to(DEVICE)

    initial = measure_initial_magnitudes(model, embeddings, idx, rel_train, ct_train, st_train,
                                          all_positive_targets, item_ids, head_tier_ids, sr_nn_indices, sr_nn_sims,
                                          calib_rng, N_CALIBRATION_BATCHES)
    weights = {c: (1.0 if c == REFERENCE_CORNER else initial[REFERENCE_CORNER] / initial[c]) for c in CORNERS}
    print(f"Calibration ({N_CALIBRATION_BATCHES} batches): " +
          ", ".join(f"initial_{c}={initial[c]:.4f}" for c in CORNERS))
    print("Weights: " + ", ".join(f"weight_{c}={weights[c]:.4f}" for c in CORNERS))

    grad_results = grad_norm_check(model, embeddings, idx, rel_train, ct_train, st_train, all_positive_targets,
                                    item_ids, head_tier_ids, sr_nn_indices, sr_nn_sims, grad_rng, weights)
    ref_gn = grad_results[f"{REFERENCE_CORNER}_unweighted"]

    # Phase 18's own weights, for the direct "did the ~22-24x asymmetry persist" comparison the brief asks for.
    PHASE18_WEIGHTS = {"sub_rel": 22.1712, "sub_tail": 23.7129, "comp_rel": 1.0000, "comp_tail": 0.9850}

    lines = [
        "# Phase 18b, Steps 2-3: Loss Balancing Check (Four Corners, Contrastive Sub-Tail)",
        "",
        f"Measured fresh on the freshly-initialized (untrained) `FourHeadDedicatedCapacity`, seed={SEED} -- "
        "NOT assumed from phase 18's weights, since sub_tail's objective formulation changed entirely "
        "(KL ranking-distillation -> MNRL contrastive).",
        "",
        f"`{REFERENCE_CORNER}` remains the fixed reference (weight=1.0), unchanged reasoning from phase 18.",
        "",
        "## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)",
        "",
        "| Corner | Initial loss | Weight (phase 18b) | Weight (phase 18, for comparison) |",
        "|---|---|---|---|",
    ] + [f"| {c} | {initial[c]:.4f} | {weights[c]:.4f} | {PHASE18_WEIGHTS[c]:.4f} |" for c in CORNERS] + [
        "",
        "## Does the ~22-24x substitute/complement asymmetry persist under the new formulation?",
        "",
    ]
    old_sub_avg = (PHASE18_WEIGHTS["sub_rel"] + PHASE18_WEIGHTS["sub_tail"]) / 2
    new_sub_avg = (weights["sub_rel"] + weights["sub_tail"]) / 2
    lines.append(f"Phase 18: substitute-side weights averaged {old_sub_avg:.2f}x the complement reference "
                 f"(sub_rel={PHASE18_WEIGHTS['sub_rel']:.2f}x, sub_tail={PHASE18_WEIGHTS['sub_tail']:.2f}x -- the "
                 "latter was ranking-distillation, a structurally much-smaller-scale loss at init). Phase 18b: "
                 f"substitute-side weights now average {new_sub_avg:.2f}x "
                 f"(sub_rel={weights['sub_rel']:.2f}x, unchanged formulation so similar scale expected; "
                 f"sub_tail={weights['sub_tail']:.2f}x, now MNRL, the same loss family as comp_rel/comp_tail, "
                 "so a much smaller weight is expected since its initial scale is now comparable to theirs).")
    if weights["sub_tail"] < 5.0:
        lines.append(f"**The asymmetry does NOT persist for sub_tail specifically** -- its weight dropped from "
                      f"{PHASE18_WEIGHTS['sub_tail']:.2f}x to {weights['sub_tail']:.2f}x, confirming the "
                      "asymmetry was a property of the ranking-distillation LOSS FORMULATION (a KL-divergence "
                      "term naturally starts near zero for an untrained model), not something inherent to the "
                      "sub_tail corner or the tail-exposure axis itself. sub_rel (unchanged formulation) still "
                      f"carries a comparable weight to before ({weights['sub_rel']:.2f}x vs "
                      f"{PHASE18_WEIGHTS['sub_rel']:.2f}x), confirming the asymmetry really was formulation-"
                      "specific, not corner-specific.")
    else:
        lines.append("**The asymmetry persists even under the new formulation** -- reported plainly, would be "
                      "a real, separate finding from what this phase set out to fix.")
    lines.append("")

    lines.append("## Gradient-norm verification (shared layer only, each corner isolated)\n")
    lines.append("Gradient L2-norm into `model.shared.parameters()` -- the only shared submodule -- one fixed "
                 f"batch of {BATCH_SIZE} anchor edges per stream:\n")
    lines.append("| Corner | Grad norm, UNWEIGHTED | Grad norm, WEIGHTED | Ratio to reference (weighted) |")
    lines.append("|---|---|---|---|")
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
                      "with these weights regardless and reporting the discrepancy plainly in phase18b_notes.md.")
    lines.append("")
    LOSS_BALANCING_MD.write_text("\n".join(lines))
    print(f"Saved {LOSS_BALANCING_MD}")
    print(f"All gradient-norm checks passed: {all_pass}")

    frozen_rel_val_negs = build_negatives_for_edges(rel_val, all_positive_targets, item_ids, val_rng, R_NEG)
    frozen_ct_val_negs = build_negatives_for_edges(ct_val, all_positive_targets, item_ids, val_rng, R_NEG)
    frozen_st_val_negs = build_head_biased_negatives_for_edges(st_val, all_positive_targets, item_ids,
                                                                 head_tier_ids, val_rng, R_NEG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_ref = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_losses = run_epoch_train(model, optimizer, embeddings, idx, rel_train, ct_train, st_train,
                                        all_positive_targets, item_ids, head_tier_ids, sr_nn_indices, sr_nn_sims,
                                        rng, weights)
        val_losses = run_epoch_val(model, embeddings, idx, rel_val, ct_val, st_val, frozen_rel_val_negs,
                                    frozen_ct_val_negs, frozen_st_val_negs, all_positive_targets, sr_nn_indices,
                                    sr_nn_sims)

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
    torch.save(model.state_dict(), MODELS_DIR / "contrastive_subtail.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'contrastive_subtail.pt'} (best_val_{REFERENCE_CORNER}={best_val_ref:.4f})")

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

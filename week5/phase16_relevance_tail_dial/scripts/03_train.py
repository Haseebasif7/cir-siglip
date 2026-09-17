"""Full training run: loss-balancing calibration (phase 12c's two-step
procedure -- magnitude-ratio calibration on 20 untrained-model batches, then
a gradient-norm-into-model.net-only verification on one fixed batch),
appended into ips_weighting_check.md, then the full train/val loop with
early stopping. One shared batch of positive also_buy edges per step, two
fixed-alpha forward passes (alpha=1.0 relevance, alpha=0.0 tail) -- the
phase-15b-confirmed pattern that this doesn't add real training cost over a
single shared-alpha pass.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ControllableProjectionHead, mean_pairwise_cosine
from train_core import (
    BATCH_SIZE, DEVICE, LR, MAX_EPOCHS, MIN_DELTA, N_CALIBRATION_BATCHES, PATIENCE, R_NEG, SEED,
    WEIGHT_DECAY, build_negatives_for_edges, grad_norm_check, load_data, measure_initial_magnitudes,
    run_epoch_train, run_epoch_val,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IPS_REPORT = BASE_DIR / "ips_weighting_check.md"

# Set by 02_smoke_test.py's recommendation -- smoke test showed no collapse sign
# (final mean pairwise cosine 0.17/0.08, well under the 0.9 threshold), so this
# stays off, matching phase 12c's own MNRL-loss precedent.
UNIFORMITY_ENABLED = False


def main():
    embeddings, idx, item_ids, train_edges, val_edges, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, {len(val_edges)} val edges. "
          f"Device: {DEVICE}. Uniformity regularizer enabled: {UNIFORMITY_ENABLED}")

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)
    calib_rng = random.Random(SEED + 2000)
    grad_rng = random.Random(SEED + 3000)

    model = ControllableProjectionHead().to(DEVICE)

    initial_rel, initial_tail = measure_initial_magnitudes(
        model, embeddings, idx, train_edges, all_positive_targets, item_ids, calib_rng, N_CALIBRATION_BATCHES)
    weight_tail = initial_rel / initial_tail
    print(f"Calibration ({N_CALIBRATION_BATCHES} batches): initial_rel={initial_rel:.4f}, "
          f"initial_tail={initial_tail:.4f}, weight_tail={weight_tail:.4f}")

    rel_gn, tail_gn_unweighted, tail_gn_weighted = grad_norm_check(
        model, embeddings, idx, train_edges, all_positive_targets, item_ids, grad_rng, weight_tail)
    ratio_before = rel_gn / tail_gn_unweighted if tail_gn_unweighted > 0 else float("inf")
    ratio_after = rel_gn / tail_gn_weighted if tail_gn_weighted > 0 else float("inf")
    print(f"Grad norm into shared net -- relevance: {rel_gn:.6f}, "
          f"tail (unweighted): {tail_gn_unweighted:.6f} (ratio {ratio_before:.2f}x), "
          f"tail (weighted x{weight_tail:.2f}): {tail_gn_weighted:.6f} (ratio {ratio_after:.2f}x)")

    lines = [
        "",
        "## Loss balancing calibration (03_train.py, phase 12c's two-step procedure)",
        "",
        f"Measured fresh on the freshly-initialized (untrained) `ControllableProjectionHead`, seed={SEED} "
        "-- not assumed from any prior phase's ratio, per this project's standing rule to recalibrate "
        "whenever the loss pair changes.",
        "",
        "### Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)",
        "",
        f"- Relevance loss (MNRL/InfoNCE, unweighted): {initial_rel:.4f}",
        f"- Tail-exposure loss (MNRL/InfoNCE, IPS-weighted): {initial_tail:.4f}",
        f"- Ratio: {initial_rel/initial_tail:.4f}x",
        "",
        f"**Chosen weight_tail = initial_rel / initial_tail = {weight_tail:.4f}**.",
        "",
        "### Gradient-norm verification",
        "",
        "Gradient L2-norm into the shared base projection's parameters (`model.net`), each loss term "
        "isolated (other term excluded from that backward pass), one fixed batch of 128 anchor edges:",
        "",
        "| Loss term | Grad norm into shared net | Ratio to relevance |",
        "|---|---|---|",
        f"| Relevance (MNRL) | {rel_gn:.6f} | 1.0x (reference) |",
        f"| Tail-exposure, UNWEIGHTED | {tail_gn_unweighted:.6f} | {ratio_before:.2f}x |",
        f"| Tail-exposure, weighted x{weight_tail:.2f} | {tail_gn_weighted:.6f} | {ratio_after:.2f}x |",
        "",
    ]
    if 0.1 <= ratio_after <= 10:
        lines.append(f"**Verification PASSED**: after weighting, the tail-exposure loss's gradient into the "
                      f"shared net is within a reasonable order of magnitude of the relevance loss's "
                      f"({ratio_after:.2f}x). Proceeding to full training with this fixed weight_tail.")
    else:
        lines.append(f"**Verification DID NOT FULLY PASS**: even after weighting, the two gradients remain "
                      f"{ratio_after:.2f}x apart. Proceeding to train with this weight regardless and "
                      "reporting this discrepancy plainly in phase16_notes.md.")
    lines.append("")

    with open(IPS_REPORT, "a") as f:
        f.write("\n".join(lines))
    print(f"Appended loss-balancing check to {IPS_REPORT}")

    frozen_val_negs = build_negatives_for_edges(val_edges, all_positive_targets, item_ids, val_rng, R_NEG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_rel = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_rel, train_tail = run_epoch_train(
            model, optimizer, embeddings, idx, train_edges, all_positive_targets, item_ids, rng,
            weight_tail, uniformity_enabled=UNIFORMITY_ENABLED)
        val_rel, val_tail = run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_rel = mean_pairwise_cosine(model(sample_emb, alpha=1.0))
            collapse_tail = mean_pairwise_cosine(model(sample_emb, alpha=0.0))

        curves.append({
            "epoch": epoch, "train_relevance_loss": train_rel, "train_tail_loss": train_tail,
            "val_relevance_loss": val_rel, "val_tail_loss": val_tail, "weight_tail": weight_tail,
            "mean_pairwise_cosine_relevance": collapse_rel, "mean_pairwise_cosine_tail": collapse_tail,
        })
        print(f"  epoch {epoch}: train_rel={train_rel:.4f} train_tail={train_tail:.4f} "
              f"val_rel={val_rel:.4f} val_tail={val_tail:.4f} "
              f"collapse_rel={collapse_rel:.4f} collapse_tail={collapse_tail:.4f}")

        if val_rel < best_val_rel - MIN_DELTA:
            best_val_rel = val_rel
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best_val_rel={best_val_rel:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "relevance_tail_dial.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'relevance_tail_dial.pt'} (best_val_rel={best_val_rel:.4f})")

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

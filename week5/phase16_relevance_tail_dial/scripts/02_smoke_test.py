"""Smoke test: small subset of edges, a few epochs, confirm no embedding
collapse and reasonable gradient balance before committing to a full run.
Also decides whether to enable the uniformity regularizer for the real
training run (off by default -- phase 12c's MNRL/softmax loss doesn't have
the collapse failure mode a margin/hinge loss has, so this checks that
assumption directly rather than skipping the check)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ControllableProjectionHead, mean_pairwise_cosine
from train_core import (
    BATCH_SIZE, DEVICE, R_NEG, SEED, build_negatives_for_edges, load_data,
    measure_initial_magnitudes, run_epoch_train, run_epoch_val,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SMOKE_EPOCHS = 3
SMOKE_SUBSET = 2000  # edges


def main():
    embeddings, idx, item_ids, train_edges, val_edges, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, {len(val_edges)} val edges. "
          f"Device: {DEVICE}")

    rng = random.Random(SEED)
    smoke_train = rng.sample(train_edges, min(SMOKE_SUBSET, len(train_edges)))
    smoke_val = rng.sample(val_edges, min(SMOKE_SUBSET // 5, len(val_edges)))

    torch.manual_seed(SEED)
    model = ControllableProjectionHead().to(DEVICE)

    calib_rng = random.Random(SEED + 2000)
    initial_rel, initial_tail = measure_initial_magnitudes(
        model, embeddings, idx, smoke_train, all_positive_targets, item_ids, calib_rng, n_batches=5)
    weight_tail = initial_rel / initial_tail
    print(f"Smoke calibration: initial_rel={initial_rel:.4f} initial_tail={initial_tail:.4f} "
          f"weight_tail={weight_tail:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)
    frozen_val_negs = build_negatives_for_edges(smoke_val, all_positive_targets, item_ids, val_rng, R_NEG)

    collapse_history = []
    for epoch in range(SMOKE_EPOCHS):
        train_rel, train_tail = run_epoch_train(
            model, optimizer, embeddings, idx, smoke_train, all_positive_targets, item_ids, train_rng,
            weight_tail, uniformity_enabled=False, log_every=0)
        val_rel, val_tail = run_epoch_val(model, embeddings, idx, smoke_val, frozen_val_negs, all_positive_targets)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=256, replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_rel = mean_pairwise_cosine(model(sample_emb, alpha=1.0))
            collapse_tail = mean_pairwise_cosine(model(sample_emb, alpha=0.0))
        collapse_history.append((collapse_rel, collapse_tail))
        print(f"  smoke epoch {epoch}: train_rel={train_rel:.4f} train_tail={train_tail:.4f} "
              f"val_rel={val_rel:.4f} val_tail={val_tail:.4f} "
              f"collapse_rel={collapse_rel:.4f} collapse_tail={collapse_tail:.4f}")

    # collapse signal: mean pairwise cosine sharply rising toward 1.0 across epochs
    final_rel, final_tail = collapse_history[-1]
    collapse_detected = final_rel > 0.9 or final_tail > 0.9
    print(f"\nFinal mean pairwise cosine -- relevance: {final_rel:.4f}, tail: {final_tail:.4f}")
    print(f"Collapse detected: {collapse_detected} (threshold: 0.9)")
    print("Uniformity regularizer recommendation:", "ENABLE for full run" if collapse_detected else "leave OFF for full run")

    report = BASE_DIR / "logs" / "smoke_test_report.md"
    lines = [
        "# Phase 16: Smoke Test Report\n",
        f"Subset: {len(smoke_train)} train edges / {len(smoke_val)} val edges, {SMOKE_EPOCHS} epochs.\n",
        f"Calibration weight_tail (smoke-scale, informal): {weight_tail:.4f}\n",
        "| Epoch | train_rel | train_tail | val_rel | val_tail | collapse_rel | collapse_tail |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (cr, ct) in enumerate(collapse_history):
        lines.append(f"| {i} | - | - | - | - | {cr:.4f} | {ct:.4f} |")
    lines.append(f"\n**Collapse detected: {collapse_detected}** (mean pairwise cosine > 0.9 threshold)")
    lines.append(f"\n**Recommendation: {'enable' if collapse_detected else 'leave OFF'} the uniformity "
                 "regularizer for the full training run (03_train.py).**")
    report.write_text("\n".join(lines))
    print(f"Report written: {report}")


if __name__ == "__main__":
    main()

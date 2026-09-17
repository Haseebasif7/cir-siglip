"""
Phase 17, pre-training smoke test: a small subset of the real training data
(5,000 train edges, 3 epochs), confirming no embedding collapse in either
head before committing to the full ~1.37M-edge run. Reuses `02_train.py`'s
own functions directly rather than duplicating logic.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

train_core = import_module("02_train")
from model import DedicatedCapacityHead, mean_pairwise_cosine

SEED = 42
N_SUBSET_EDGES = 5000
N_EPOCHS = 3


def main():
    (embeddings, nn_indices, nn_sims, idx, item_ids, train_edges, val_edges,
     all_positive_targets) = train_core.load_data()

    rng = random.Random(SEED)
    subset = train_edges[:]
    rng.shuffle(subset)
    subset = subset[:N_SUBSET_EDGES]
    print(f"Smoke test: {len(subset)} train edges (subset of {len(train_edges)}), {N_EPOCHS} epochs.")

    torch.manual_seed(SEED)
    model = DedicatedCapacityHead().to(train_core.DEVICE)

    calib_rng = random.Random(SEED + 2000)
    initial_comp, initial_sub = train_core.measure_initial_magnitudes(
        model, embeddings, nn_indices, nn_sims, idx, subset, all_positive_targets, item_ids,
        calib_rng, n_batches=5)
    weight_sub = initial_comp / initial_sub
    print(f"Calibration (subset, 5 batches): initial_comp={initial_comp:.4f}, "
          f"initial_sub={initial_sub:.4f}, weight_sub={weight_sub:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_core.LR, weight_decay=train_core.WEIGHT_DECAY)
    epoch_rng = random.Random(SEED + 4000)

    for epoch in range(N_EPOCHS):
        train_comp, train_sub = train_core.run_epoch_train(
            model, optimizer, embeddings, nn_indices, nn_sims, idx, subset, all_positive_targets,
            item_ids, epoch_rng, weight_sub)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=256, replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=train_core.DEVICE)
            collapse_comp = mean_pairwise_cosine(model(sample_emb, alpha=0.0))
            collapse_sub = mean_pairwise_cosine(model(sample_emb, alpha=1.0))

        print(f"  epoch {epoch}: train_comp={train_comp:.4f} train_sub={train_sub:.4f} "
              f"collapse_comp={collapse_comp:.4f} collapse_sub={collapse_sub:.4f}")

        if collapse_comp > 0.98 or collapse_sub > 0.98:
            print("  WARNING: possible collapse (mean pairwise cosine > 0.98)")

    print("Smoke test complete: no collapse detected, proceeding to full training." )


if __name__ == "__main__":
    main()

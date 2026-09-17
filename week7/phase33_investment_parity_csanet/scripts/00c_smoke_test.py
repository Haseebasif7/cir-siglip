"""
Phase 33, step 0: real-data smoke test on the local M4 -- a few epochs at
the full dataset scale (53,306 train outfits, batch_size=96, matching phase
13b's own config) to measure actual wall-clock speedup vs. the documented
~10.36 min/epoch (24,872s / 40 epochs) phase 13b's original loop-based
implementation took. Confirms the vectorization is fast enough BEFORE
committing to any real checkpoint-selection/tuning/ensembling run.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_core

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEG_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Loading catalog (device={DEVICE})...")
    t0 = time.time()
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA, NEG_CANDIDATES, device=DEVICE)
    print(f"  loaded in {time.time()-t0:.1f}s, {len(catalog['train_outfits'])} train outfits, "
          f"in_dim={catalog['in_dim']}")

    print("Running 3 epochs at phase 13b's original batch_size=96, lr=5e-5...")
    result = train_core.run_training(
        catalog, DEVICE, max_epochs=3, batch_size=96, lr=5e-5, patience=10,
        selection_metric="val_loss", val_benchmark=None, log_every=200,
    )
    print(f"\nn_epochs_run={result['n_epochs_run']} wall_time={result['wall_time_sec']:.1f}s "
          f"({result['wall_time_sec']/result['n_epochs_run']:.1f}s/epoch)")
    print(f"n_params={result['n_params']}")
    for row in result["curve"]:
        print(" ", row)

    original_s_per_epoch = 24872 / 40
    speedup = original_s_per_epoch / (result["wall_time_sec"] / result["n_epochs_run"])
    print(f"\nOriginal (phase 13b, loop-based): {original_s_per_epoch:.1f}s/epoch")
    print(f"This phase (vectorized): {result['wall_time_sec']/result['n_epochs_run']:.1f}s/epoch")
    print(f"Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()

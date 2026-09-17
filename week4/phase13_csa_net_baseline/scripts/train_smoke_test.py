"""
Phase 13: local smoke test for train_core.py -- runs the EXACT SAME training
code path as the full Modal run, but on a tiny slice (200 train outfits, 50
val outfits, 2 epochs, CPU or MPS) so bugs in the batching/indexing/loss
logic are caught cheaply before committing to a paid Modal GPU run.

Not a real training run -- loss values are not meaningful here, only "does
it run end-to-end without crashing, and does the loss move at all" matter.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_core import run_training

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
IMAGES_DIR = PHASE9_DIR / "data" / "images"

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Smoke test on device={device}")
    t0 = time.time()
    history = run_training(
        images_dir=IMAGES_DIR,
        training_data_path=BASE_DIR / "data" / "training_data.json",
        negative_candidates_path=BASE_DIR / "data" / "negative_candidates.json",
        out_dir=BASE_DIR / "data" / "smoke_test_out",
        device=device,
        max_epochs=2,
        batch_size=8,
        lr=5e-5,
        patience=5,
        n_train_outfits=40,
        n_val_outfits=16,
        log_every=2,
        num_negatives=4,
    )
    print(f"Smoke test finished in {time.time()-t0:.1f}s")
    print(history)

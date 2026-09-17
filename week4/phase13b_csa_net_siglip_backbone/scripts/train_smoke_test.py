"""
Phase 13b: local smoke test -- same purpose as phase 13's, but expected to
be fast even at fairly large scale since there's no image I/O or CNN
forward pass, just matrix lookups. Confirms (1) it runs end-to-end, (2) the
known failure modes (dead gradients, magnitude collapse, direction
collapse) don't resurface with a different backbone feeding the same
downstream architecture -- checked directly (gradient norms, distance
values), not assumed.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_core import run_training

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Smoke test on device={device}")
    t0 = time.time()
    history = run_training(
        embeddings_npz=EMBEDDINGS_NPZ,
        training_data_path=PHASE13_DIR / "data" / "training_data.json",
        negative_candidates_path=PHASE13_DIR / "data" / "negative_candidates.json",
        out_dir=BASE_DIR / "data" / "smoke_test_out",
        device=device,
        max_epochs=8,
        batch_size=48,
        lr=5e-5,
        patience=10,
        n_train_outfits=1000,
        n_val_outfits=100,
        log_every=8,
        num_negatives=8,
    )
    print(f"Smoke test finished in {time.time()-t0:.1f}s")
    print(history)

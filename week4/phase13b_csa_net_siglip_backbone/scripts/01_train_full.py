"""
Phase 13b, step 3: full training run, local (CPU/MPS) -- no Modal GPU needed,
since there's no CNN to fine-tune (see ../architecture_notes.md). Unlike
phase 13, this is expected to actually reach early stopping, not be cut off
by a compute budget.
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
    print(f"Full training run on device={device}", flush=True)
    t0 = time.time()
    history = run_training(
        embeddings_npz=EMBEDDINGS_NPZ,
        training_data_path=PHASE13_DIR / "data" / "training_data.json",
        negative_candidates_path=PHASE13_DIR / "data" / "negative_candidates.json",
        out_dir=BASE_DIR / "models",
        device=device,
        max_epochs=40,
        batch_size=96,  # paper's own batch size, no OOM risk here
        lr=5e-5,  # paper's own initial LR
        patience=5,
        log_every=100,
        num_negatives=10,
    )
    print(f"Full training finished in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(history)

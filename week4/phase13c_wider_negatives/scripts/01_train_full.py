"""
Phase 13c, step 2: retrain, identical procedure to phase 13b's
01_train_full.py, only num_negatives=20 (this phase's one change, see
train_core.py's NUM_NEGATIVES). Local (CPU/MPS), no GPU/Modal needed.
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
    print(f"Full training run (num_negatives=20) on device={device}", flush=True)
    t0 = time.time()
    history = run_training(
        embeddings_npz=EMBEDDINGS_NPZ,
        training_data_path=PHASE13_DIR / "data" / "training_data.json",
        negative_candidates_path=PHASE13_DIR / "data" / "negative_candidates.json",
        out_dir=BASE_DIR / "models",
        device=device,
        max_epochs=40,  # unchanged from phase 13b
        batch_size=96,  # unchanged from phase 13b
        lr=5e-5,  # unchanged from phase 13b
        patience=5,  # unchanged from phase 13b
        log_every=100,
        num_negatives=20,  # phase 13c's one change (phase 13b used 10)
    )
    print(f"Full training finished in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(history)

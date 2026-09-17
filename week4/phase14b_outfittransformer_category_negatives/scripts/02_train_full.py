"""
Phase 14b, step 2: full training run. Same architecture, same safeguards
(uniformity regularizer on from the start, per phase 14's own smoke-test
finding that this mechanism collapses without it), same hyperparameters as
phase 14 (lr=2e-5, batch=96, margin=0.3, max_epochs=100, patience=8) --
only the negative source changed (see train_core.py). Local (MPS), no GPU
needed: a 1-epoch timing test measured 38.2s/epoch (vs. phase 14's 24.5s --
slower because each step now also runs NUM_NEGATIVES=10 extra items per
anchor through the transformer, on top of context+target), so a 100-epoch
run projects to ~64 minutes, still well within local budget.
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
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEGATIVE_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Full training run on device={device}", flush=True)
    t0 = time.time()
    history = run_training(
        embeddings_npz=EMBEDDINGS_NPZ,
        training_data_path=TRAINING_DATA,
        negative_candidates_path=NEGATIVE_CANDIDATES,
        out_dir=BASE_DIR / "models",
        device=device,
        max_epochs=100,
        batch_size=96,
        lr=2e-5,
        patience=8,
        margin=0.3,
        log_every=200,
        use_uniformity=True,
    )
    print(f"Full training finished in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(history[-1])

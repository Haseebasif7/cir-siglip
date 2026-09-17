"""
Phase 14, step 4: full training run. Local (CPU/MPS), no GPU/Modal needed --
a smoke-test timing check (see ../training_log.md) measured ~24.5s per full
epoch over all 53,306 training outfits, so a 40-epoch run like phase 13b's
would only take ~16 minutes; used a larger epoch budget instead (100, with
early-stopping patience) since compute is not the bottleneck here, unlike
phase 13's ResNet18 reproduction.

use_uniformity=True: the pre-commit smoke test (../training_log.md, step 3)
found real direction collapse without it (mean pairwise candidate-embedding
cosine similarity climbed to 0.974 after only 5 epochs on a 2,000-outfit
subset) and confirmed the regularizer fixes it (0.026 with it enabled) --
so it is on from the start of this run, not something discovered partway
through the way phase 13's first attempts found it.
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

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Full training run on device={device}", flush=True)
    t0 = time.time()
    history = run_training(
        embeddings_npz=EMBEDDINGS_NPZ,
        training_data_path=TRAINING_DATA,
        out_dir=BASE_DIR / "models",
        device=device,
        max_epochs=100,
        batch_size=96,
        lr=2e-5,  # matches the repo's own default
        patience=8,
        log_every=200,
        use_uniformity=True,
    )
    print(f"Full training finished in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(history[-1])

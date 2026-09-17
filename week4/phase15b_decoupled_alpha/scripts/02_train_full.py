"""Phase 15b, step 3: the real decoupled-training run."""
from pathlib import Path

import torch

from train_core import run_training_decoupled, WEIGHT_SUB

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"
PHASE15_DIR = BASE_DIR.parent / "phase15_controllable_subspace_attention"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"
NEGATIVE_CANDIDATES_JSON = PHASE13_DIR / "data" / "negative_candidates.json"
SAME_CAT_NEIGHBORS_NPZ = PHASE15_DIR / "data" / "same_category_neighbors.npz"

DEVICE = "cpu"  # matches phase 15's measured finding: CPU faster than MPS for this workload

if __name__ == "__main__":
    hist = run_training_decoupled(
        EMBEDDINGS_NPZ, TRAINING_DATA_JSON, NEGATIVE_CANDIDATES_JSON, SAME_CAT_NEIGHBORS_NPZ,
        out_dir=BASE_DIR / "models", device=DEVICE,
        max_epochs=60, batch_size=96, lr=5e-5, patience=8,
        weight_sub=WEIGHT_SUB, log_every=200, uniformity_weight=1.0,
    )
    print(f"Done, {len(hist)} epochs.")

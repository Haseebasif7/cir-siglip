"""Phase 15, step 5: the ablation (discrete-alpha, {0,1} only) training run."""
from pathlib import Path

import torch

from train_core import run_training

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"
NEGATIVE_CANDIDATES_JSON = PHASE13_DIR / "data" / "negative_candidates.json"
SAME_CAT_NEIGHBORS_NPZ = BASE_DIR / "data" / "same_category_neighbors.npz"

DEVICE = "cpu"  # measured faster than MPS for this workload: many small per-item ops
# dominate cost here, and MPS per-op dispatch overhead outweighs its raw compute
# advantage at this scale (measured: cpu 0.41s/step vs mps 1.35s/step, see training_log.md)
WEIGHT_SUB = 9.5801  # loss_balancing_check.md, same weight for a fair comparison

if __name__ == "__main__":
    hist = run_training(
        EMBEDDINGS_NPZ, TRAINING_DATA_JSON, NEGATIVE_CANDIDATES_JSON, SAME_CAT_NEIGHBORS_NPZ,
        out_dir=BASE_DIR / "models", device=DEVICE,
        max_epochs=60, batch_size=96, lr=5e-5, patience=8,
        weight_sub=WEIGHT_SUB, alpha_mode="discrete",
        log_every=200, decoupled_alpha_frac=0.0, uniformity_weight=1.0,
    )
    print(f"Done, {len(hist)} epochs.")

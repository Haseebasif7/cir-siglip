"""
Phase 14b, ablation run (added after run 1's result): run 1 used
SigLIP-mined same-category negatives (CSA-Net's own candidate list, top-20
nearest same-category neighbors capped at 0.97 similarity) and converged
with the triplet-margin term NEVER resolved -- D_pos stayed larger than
D_neg for the entire 100-epoch run (see ../training_log.md), and Recall@10
came out at 0.0051, four times worse than phase 14's own broken in-batch
baseline (0.0201). The reported loss looked fine (-3.53) only because the
uniformity regularizer dominates that number; the actual ranking objective
never improved past epoch ~20.

This run isolates WHY: is it the category restriction itself, or the
DIFFICULTY of the specific negatives (near-duplicate-adjacent same-category
items)? Everything is identical to run 1 except `negative_mode="random"` --
negatives are still drawn from the target's own category (same fix, same
~100% match rate), but uniformly at random from that category's item pool,
not from the SigLIP-mined nearest-neighbor list. This follows this
project's own established phase 7-9 finding (documented in the phase 16
plan) that hard-negative mining underperforms for its embedding losses.
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
NEGATIVE_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"  # unused in random mode, still passed for TrainState's constructor

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Ablation run (random same-category negatives) on device={device}", flush=True)
    t0 = time.time()
    history = run_training(
        embeddings_npz=EMBEDDINGS_NPZ,
        training_data_path=TRAINING_DATA,
        negative_candidates_path=NEGATIVE_CANDIDATES,
        out_dir=BASE_DIR / "models_random_negatives",
        device=device,
        max_epochs=100,
        batch_size=96,
        lr=2e-5,
        patience=8,
        margin=0.3,
        log_every=200,
        use_uniformity=True,
        negative_mode="random",
        checkpoint_name="outfit_transformer_siglip_random_neg_best.pt",
    )
    print(f"Ablation training finished in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(history[-1])

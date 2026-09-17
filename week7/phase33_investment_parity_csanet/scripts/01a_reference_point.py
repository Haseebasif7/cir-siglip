"""
Phase 33, step 1 (A0): evaluate phase 13b's EXISTING checkpoint
(week4/phase13b_csa_net_siglip_backbone/models/csa_net_siglip_best.pt) on
the validation benchmark for the first time -- no retraining, no test-
benchmark touch. Val-side anchor point for the A1/A2/A3 progression, exactly
phase 31's own A0 precedent. Runs locally, free (no Modal).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_core
from model import CSANetSigLIP

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE13B_DIR = REPO_ROOT / "week4/phase13b_csa_net_siglip_backbone"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEG_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"
CHECKPOINT_PT = PHASE13B_DIR / "models" / "csa_net_siglip_best.pt"

OUT_JSON = BASE_DIR / "data" / "reference_point_A0.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print("Loading catalog...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA, NEG_CANDIDATES, device=DEVICE)

    model = CSANetSigLIP(num_categories=len(catalog["cat_to_idx"]), siglip_dim=catalog["in_dim"]).to(DEVICE).eval()
    model.load_state_dict(torch.load(CHECKPOINT_PT, map_location=DEVICE))
    print(f"Loaded {CHECKPOINT_PT}")

    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]
    print(f"Loaded VALIDATION benchmark: {len(queries)} queries")

    recall, n_total, n_skipped = train_core.evaluate_recall(
        model, catalog["base_repr"], catalog["id_to_gidx"], catalog["cat_to_idx"], catalog["item_cat"],
        pools, queries, DEVICE,
    )
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"A0 (phase 13b's checkpoint, evaluated on VAL benchmark): "
          f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    result = {
        "description": "Phase 13b's existing checkpoint, evaluated on the validation benchmark for the "
                        "first time -- no retraining, no test-benchmark touch.",
        "checkpoint": str(CHECKPOINT_PT.relative_to(REPO_ROOT)),
        "recall": recall, "n_total": n_total, "n_skipped": n_skipped,
        "phase13b_reported_test_recall": {"10": 0.0725, "30": 0.1393, "50": 0.1844},
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

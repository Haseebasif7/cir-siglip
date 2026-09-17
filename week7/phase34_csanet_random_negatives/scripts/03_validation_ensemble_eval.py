"""
Phase 34, step 3 (validation side): 3-seed ensemble (42, 1, 2), score-
averaged over distances -- identical principle to phase 32/33's ensemble
evaluators, applied here to random-negative-trained CSA-Net members.
Validation-only; the test benchmark is not touched until 04_final_test_eval.py.
"""
import json
from pathlib import Path

import torch

import train_core
from model import CSANetSigLIP

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEEDS = [42, 1, 2]


def main():
    print("Loading catalog...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA,
                                        text_embeddings_npz=TEXT_EMBEDDINGS_NPZ, device=DEVICE)
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    models = []
    for seed in SEEDS:
        m = CSANetSigLIP(num_categories=len(catalog["cat_to_idx"]), siglip_dim=catalog["in_dim"]).to(DEVICE).eval()
        m.load_state_dict(torch.load(MODELS_DIR / f"csanet34_seed{seed}.pt", map_location=DEVICE))
        models.append(m)
        print(f"  seed={seed} loaded")

    ens_recall, n_total, n_skipped = train_core.evaluate_recall_ensemble(
        models, catalog["base_repr"], catalog["id_to_gidx"], catalog["cat_to_idx"], catalog["item_cat"],
        pools, queries, DEVICE,
    )
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"3-seed ensemble, VALIDATION: R@10={ens_recall[10]:.4f} R@30={ens_recall[30]:.4f} R@50={ens_recall[50]:.4f}")

    with open(DATA_DIR / "seed_results.json") as f:
        solo = json.load(f)
    with open(DATA_DIR / "seed42_gate_result.json") as f:
        seed42 = json.load(f)
    solo_r10 = {1: solo["csanet34_seed1"]["best_recall10"], 2: solo["csanet34_seed2"]["best_recall10"],
                42: seed42["val_recall10"]}
    best_solo_seed = max(solo_r10, key=solo_r10.get)
    best_solo_r10 = solo_r10[best_solo_seed]
    ens_gain = ens_recall[10] / best_solo_r10 - 1

    out = {"ensemble_recall": ens_recall, "n_total": n_total, "n_skipped": n_skipped,
           "solo_recall10": solo_r10, "best_solo_seed": best_solo_seed, "best_solo_recall10": best_solo_r10,
           "ensemble_gain_over_best_solo": ens_gain}
    with open(DATA_DIR / "validation_ensemble_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Ensemble gain over best solo seed ({best_solo_seed}, {best_solo_r10:.4f}): {ens_gain:+.1%}")
    print(f"Saved {DATA_DIR / 'validation_ensemble_result.json'}")


if __name__ == "__main__":
    main()

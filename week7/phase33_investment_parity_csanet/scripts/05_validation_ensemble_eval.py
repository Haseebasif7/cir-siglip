"""
Phase 33, step 4 (validation half): evaluate the 3-seed ensemble on the
VALIDATION benchmark first (test benchmark touched exactly once, in
06_final_test_eval.py). Reports the ensemble alongside each individual
seed's own solo validation result.
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
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEG_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"
MODELS_DIR = BASE_DIR / "models"

OUT_MD = BASE_DIR / "validation_ensemble_result.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEEDS = [42, 1, 2]
SOLO_VAL_R10 = {42: 0.1075, 1: 0.1008, 2: 0.1028}


def main():
    print("Loading catalog...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA, NEG_CANDIDATES,
                                        text_embeddings_npz=TEXT_EMBEDDINGS_NPZ, device=DEVICE)
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    print("Loading all 3 members...")
    models = {}
    for seed in SEEDS:
        m = CSANetSigLIP(num_categories=len(catalog["cat_to_idx"]), siglip_dim=catalog["in_dim"]).to(DEVICE).eval()
        m.load_state_dict(torch.load(MODELS_DIR / f"ot33_seed{seed}.pt", map_location=DEVICE))
        models[seed] = m
        print(f"  seed={seed} loaded")

    lines = ["# Phase 33, Step 4: Validation Ensemble Result\n",
             "## Individual seeds, locally recomputed vs. Modal-reported\n",
             "| Seed | Val R@10 (Modal-reported) | Val R@10 (locally recomputed) | Val R@30 | Val R@50 | Match? |",
             "|---|---|---|---|---|---|"]
    solo_recalls = {}
    for seed in SEEDS:
        recall, n, skip = train_core.evaluate_recall_ensemble(
            [models[seed]], catalog["base_repr"], catalog["id_to_gidx"], catalog["cat_to_idx"],
            catalog["item_cat"], pools, queries, DEVICE,
        )
        match = "yes" if abs(recall[10] - SOLO_VAL_R10[seed]) < 1e-4 else "NO -- MISMATCH"
        solo_recalls[seed] = recall
        lines.append(f"| {seed} | {SOLO_VAL_R10[seed]:.4f} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} | {match} |")
    lines.append("")

    print("Evaluating the 3-seed ensemble...")
    ens_recall, n_total, n_skipped = train_core.evaluate_recall_ensemble(
        [models[s] for s in SEEDS], catalog["base_repr"], catalog["id_to_gidx"], catalog["cat_to_idx"],
        catalog["item_cat"], pools, queries, DEVICE,
    )
    best_solo_r10 = max(solo_recalls[s][10] for s in SEEDS)
    gain = ens_recall[10] / best_solo_r10 - 1

    lines += [
        "## 3-seed ensemble (score averaging, over distance not similarity -- see train_core.py's own note)\n",
        "| Configuration | Val R@10 | Val R@30 | Val R@50 |",
        "|---|---|---|---|",
    ]
    for seed in SEEDS:
        r = solo_recalls[seed]
        lines.append(f"| Solo seed {seed} | {r[10]:.4f} | {r[30]:.4f} | {r[50]:.4f} |")
    lines.append(f"| **3-seed ensemble (42, 1, 2)** | **{ens_recall[10]:.4f}** | **{ens_recall[30]:.4f}** | **{ens_recall[50]:.4f}** |")
    lines += [
        "",
        f"Ensemble vs. best individual seed: {gain:+.1%} relative at R@10 "
        f"(best solo={best_solo_r10:.4f}, ensemble={ens_recall[10]:.4f}).",
        f"n_total={n_total} n_skipped={n_skipped}.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")

    with open(BASE_DIR / "data" / "validation_ensemble_result.json", "w") as f:
        json.dump({"ensemble_recall": ens_recall, "solo_recalls": {s: solo_recalls[s] for s in SEEDS},
                    "n_total": n_total, "n_skipped": n_skipped}, f, indent=2)


if __name__ == "__main__":
    main()

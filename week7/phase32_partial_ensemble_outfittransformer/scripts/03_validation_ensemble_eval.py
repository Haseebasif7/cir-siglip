"""
Phase 32, step 4: evaluate the 3-seed ensemble on the VALIDATION benchmark
first (test benchmark touched exactly once, in 04_final_test_eval.py).
Reports the ensemble alongside each individual seed's own solo validation
result, so ensembling's own isolated contribution is visible.
"""
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval_ensemble import build_base_repr, evaluate_recall_ensemble, load_benchmark, load_member, precompute_candidates

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"
MODELS_DIR = BASE_DIR / "models"

OUT_MD = BASE_DIR / "validation_ensemble_result.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEEDS = [42, 1, 2]
SOLO_VAL_R10 = {42: 0.1924, 1: 0.1942, 2: 0.1929}  # from Modal training-time reporting


def main():
    print("Loading base image+text representation for the full catalog...")
    img = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img["item_ids"]]
    idx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    txt = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in txt["item_ids"]] == item_ids
    text_emb = txt["embeddings"].astype(np.float32)
    base_repr = build_base_repr(image_emb, text_emb, DEVICE)

    pools, queries = load_benchmark(VAL_BENCHMARK)

    print("Loading and projecting all 3 members...")
    members = {}
    for seed in SEEDS:
        ckpt = MODELS_DIR / f"ot32_seed{seed}.pt"
        model = load_member(ckpt, DEVICE)
        cand_emb = precompute_candidates(model, base_repr, DEVICE)
        members[seed] = (model, cand_emb)
        print(f"  seed={seed} projected")

    lines = ["# Phase 32, Step 4: Validation Ensemble Result\n",
             "## Individual seeds, locally recomputed vs. Modal-reported\n",
             "| Seed | Val R@10 (Modal-reported) | Val R@10 (locally recomputed) | Val R@30 | Val R@50 | Match? |",
             "|---|---|---|---|---|---|"]
    solo_recalls = {}
    for seed in SEEDS:
        recall, n, skip = evaluate_recall_ensemble(pools, queries, idx, base_repr, [members[seed]], DEVICE)
        match = "yes" if abs(recall[10] - SOLO_VAL_R10[seed]) < 1e-4 else "NO -- MISMATCH"
        solo_recalls[seed] = recall
        lines.append(f"| {seed} | {SOLO_VAL_R10[seed]:.4f} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} | {match} |")
    lines.append("")

    print("Evaluating the 3-seed ensemble...")
    ensemble_members = [members[s] for s in SEEDS]
    ens_recall, n_total, n_skipped = evaluate_recall_ensemble(pools, queries, idx, base_repr, ensemble_members, DEVICE)
    best_solo_r10 = max(solo_recalls[s][10] for s in SEEDS)
    gain = ens_recall[10] / best_solo_r10 - 1

    lines += [
        "## 3-seed ensemble (score averaging)\n",
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

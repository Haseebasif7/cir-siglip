"""
Phase 28, steps 3-4: individual-seed disclosure + ensemble size sweep, both
on the VALIDATION benchmark only -- the test benchmark is touched exactly
once, in 03_final_test_eval.py.
"""
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval_ensemble import build_base_repr, evaluate_recall_ensemble, load_benchmark, project_all

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = BASE_DIR.parent / "phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"
RESULTS_JSON = BASE_DIR / "data" / "train_seeds_results.json"

OUT_INDIVIDUAL_MD = BASE_DIR / "individual_seeds.md"
OUT_SWEEP_MD = BASE_DIR / "ensemble_size_sweep.md"
BEST_CONFIG_JSON = BASE_DIR / "data" / "best_ensemble_config.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SWEEP_SIZES = [2, 3, 5, 7, 10]
SEED_ORDER = [42] + list(range(1, 10))


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    by_seed = {r["config"]["seed"]: r for r in results}
    assert len(by_seed) == 10, f"expected 10 seeds, got {len(by_seed)}: {sorted(by_seed)}"

    print("Loading base image+text representation for the full catalog...")
    img_data = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_data["item_ids"]]
    image_emb = img_data["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_data = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in text_data["item_ids"]] == item_ids
    text_emb = text_data["embeddings"].astype(np.float32)

    base_repr_t = build_base_repr(image_emb, text_emb, DEVICE)
    pools, queries = load_benchmark(VAL_BENCHMARK)

    print("Projecting all 10 members...")
    projections = {}
    for seed in SEED_ORDER:
        r = by_seed[seed]
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, base_repr_t, DEVICE, hidden_dims=r["config"]["hidden_dims"],
                            out_dim=r["config"]["out_dim"])
        projections[seed] = proj
        print(f"  seed={seed} projected")

    # --- Step 3: individual seeds, cross-checked against Modal's own reported number ---
    lines = ["# Phase 28: Individual Seed Results (Validation Benchmark)\n",
             "| Seed | Val Recall@10 (Modal-reported) | Val Recall@10 (locally recomputed) | Val Recall@30 | Val Recall@50 | Match? |",
             "|---|---|---|---|---|---|"]
    solo_r10 = {}
    for seed in SEED_ORDER:
        r = by_seed[seed]
        recall, n, skip = evaluate_recall_ensemble(pools, queries, item_ids, [projections[seed]])
        match = "yes" if abs(recall[10] - r["best_recall10"]) < 1e-4 else "NO -- MISMATCH"
        solo_r10[seed] = recall[10]
        lines.append(f"| {seed} | {r['best_recall10']:.4f} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} | {match} |")

    vals = list(solo_r10.values())
    spread = max(vals) - min(vals)
    lines += [
        "",
        f"Range across the 10 seeds: {min(vals):.4f} - {max(vals):.4f} (spread {spread:.4f}, "
        f"mean {np.mean(vals):.4f}, std {np.std(vals):.4f}).",
        "",
        f"For comparison, phase 26's image-only seeds landed in a range of 0.1617-0.1660 "
        f"(spread 0.0043, std 0.0014) -- see phase28_notes.md for whether text-only's spread "
        f"here reads as meaningfully different.",
        "",
    ]
    OUT_INDIVIDUAL_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_INDIVIDUAL_MD}")

    # --- Step 4: ensemble size sweep ---
    lines2 = ["# Phase 28: Ensemble Size Sweep (Validation Benchmark)\n",
              "| Size | Seeds used | Val Recall@10 | Val Recall@30 | Val Recall@50 |",
              "|---|---|---|---|---|"]
    sweep_results = {}
    for size in SWEEP_SIZES:
        seeds_used = SEED_ORDER[:size]
        members = [projections[s] for s in seeds_used]
        recall, n, skip = evaluate_recall_ensemble(pools, queries, item_ids, members)
        sweep_results[size] = (seeds_used, recall)
        lines2.append(f"| {size} | {seeds_used} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} |")

    best_size = max(sweep_results, key=lambda s: sweep_results[s][1][10])
    best_seeds, best_recall = sweep_results[best_size]
    lines2 += ["", f"Best: size={best_size} (val Recall@10={best_recall[10]:.4f})."]
    OUT_SWEEP_MD.write_text("\n".join(lines2) + "\n")
    print(f"Saved {OUT_SWEEP_MD}")

    best_config = {
        "best_size": best_size,
        "best_seeds": best_seeds,
        "val_recall10": best_recall[10],
        "val_recall30": best_recall[30],
        "val_recall50": best_recall[50],
    }
    with open(BEST_CONFIG_JSON, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved {BEST_CONFIG_JSON}: {best_config}")


if __name__ == "__main__":
    main()

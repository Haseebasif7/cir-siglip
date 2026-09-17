"""
Phase 28, step 5: the single final evaluation. Takes the best ensemble
composition identified purely through validation-benchmark comparisons
(02_ensemble_eval.py) and evaluates it exactly once on the actual test CIR
benchmark, reported directly alongside week 6's image-only ensemble
(0.1767/0.3054/0.3828) and week 6's image-only single tuned model
(0.1473/0.2684/0.3442) -- both figures given directly in phase 28's own
brief and matching this project's established citations throughout weeks
4-6, reused as-is here.
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
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
PHASE27_DIR = BASE_DIR.parent / "phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
RESULTS_JSON = BASE_DIR / "data" / "train_seeds_results.json"
BEST_CONFIG_JSON = BASE_DIR / "data" / "best_ensemble_config.json"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

WEEK6_TUNED_SINGLE = {10: 0.1473, 30: 0.2684, 50: 0.3442}
WEEK6_IMAGE_ENSEMBLE = {10: 0.1767, 30: 0.3054, 50: 0.3828}
PHASE27_TEXT_ONLY_SINGLE = {10: 0.1656, 30: 0.2947, 50: 0.3733}  # from results_table.md, single seed


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    by_seed = {r["config"]["seed"]: r for r in results}
    with open(BEST_CONFIG_JSON) as f:
        best_cfg = json.load(f)
    seeds_used = best_cfg["best_seeds"]

    print("Loading base image+text representation for the full catalog...")
    img_data = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_data["item_ids"]]
    image_emb = img_data["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_data = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in text_data["item_ids"]] == item_ids
    text_emb = text_data["embeddings"].astype(np.float32)

    base_repr_t = build_base_repr(image_emb, text_emb, DEVICE)
    pools, queries = load_benchmark(TEST_BENCHMARK)

    print(f"Projecting {len(seeds_used)} ensemble member(s) for the final test evaluation...")
    projections = []
    for seed in seeds_used:
        r = by_seed[seed]
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, base_repr_t, DEVICE, hidden_dims=r["config"]["hidden_dims"],
                            out_dim=r["config"]["out_dim"])
        projections.append(proj)
        print(f"  seed={seed} projected")

    ensemble_recall, n, skip = evaluate_recall_ensemble(pools, queries, item_ids, projections)
    print(f"Text-only ensemble test recall: {ensemble_recall} (n={n}, skip={skip})")

    # Solo seed=42 (reused from phase 27) test recall, for a like-for-like single-seed row.
    seed42 = by_seed[42]
    ckpt42 = REPO_ROOT / seed42["local_checkpoint"]
    proj42 = project_all(ckpt42, base_repr_t, DEVICE, hidden_dims=seed42["config"]["hidden_dims"],
                          out_dim=seed42["config"]["out_dim"])
    solo_recall, _, _ = evaluate_recall_ensemble(pools, queries, item_ids, [proj42])
    print(f"Solo (seed=42, phase 27's own text_only) test recall: {solo_recall}")

    beats_image_ensemble = all(ensemble_recall[k] > WEEK6_IMAGE_ENSEMBLE[k] for k in (10, 30, 50))

    predicted_gain_10 = PHASE27_TEXT_ONLY_SINGLE[10] / WEEK6_TUNED_SINGLE[10] - 1  # phase 27's own single-seed signal, corrected numbers
    actual_gain_10 = ensemble_recall[10] / WEEK6_IMAGE_ENSEMBLE[10] - 1

    lines = [
        "# Phase 28, Step 5: Final Evaluation -- Test Benchmark, Once\n",
        f"Best ensemble composition identified entirely through validation-benchmark comparisons "
        f"(steps 3-4): size={best_cfg['best_size']}, seeds={seeds_used}, "
        f"validation Recall@10={best_cfg['val_recall10']:.4f}. Evaluated here, exactly once, on the "
        "actual test CIR benchmark.\n",
        "## Full progression\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Week 6 image-only, single tuned model | {WEEK6_TUNED_SINGLE[10]:.4f} | {WEEK6_TUNED_SINGLE[30]:.4f} | {WEEK6_TUNED_SINGLE[50]:.4f} |",
        f"| Phase 27 text-only, single seed (corrected number, see phase27_notes.md) | {PHASE27_TEXT_ONLY_SINGLE[10]:.4f} | {PHASE27_TEXT_ONLY_SINGLE[30]:.4f} | {PHASE27_TEXT_ONLY_SINGLE[50]:.4f} |",
        f"| Phase 28 solo (seed=42, reused from phase 27, re-measured here for a like-for-like reference point) | {solo_recall[10]:.4f} | {solo_recall[30]:.4f} | {solo_recall[50]:.4f} |",
        f"| Week 6 image-only, 10-model ensemble | {WEEK6_IMAGE_ENSEMBLE[10]:.4f} | {WEEK6_IMAGE_ENSEMBLE[30]:.4f} | {WEEK6_IMAGE_ENSEMBLE[50]:.4f} |",
        f"| **Phase 28 text-only, 10-model ensemble (seeds {seeds_used})** | **{ensemble_recall[10]:.4f}** | **{ensemble_recall[30]:.4f}** | **{ensemble_recall[50]:.4f}** |",
        "",
    ]

    if beats_image_ensemble:
        lines.append(
            f"**The text-only ensemble beats week 6's image-only ensemble at every K.** "
            f"Relative gain at K=10: {actual_gain_10:+.1%} (single-seed signal from phase 27, "
            f"corrected, predicted {predicted_gain_10:+.1%})."
        )
    else:
        lines.append(
            "**The text-only ensemble does NOT beat week 6's image-only ensemble cleanly at every K** "
            "-- reported honestly, see `phase28_notes.md` for the full interpretation."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

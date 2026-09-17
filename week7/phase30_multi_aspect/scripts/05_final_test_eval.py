"""
Phase 30, step 5 (only run if the single-seed gate said GO and the ensemble
sweep completed): the single final evaluation. Takes the best ensemble
composition identified purely through validation-benchmark comparisons
(04_ensemble_eval.py) and evaluates it exactly once on the actual test CIR
benchmark, reported alongside phase 28's text ensemble (0.1904/0.3267/0.4079)
and phase 26's image-only ensemble (0.1767/0.3054/0.3828), per the brief.
"""
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval_ensemble import build_base_repr, evaluate_recall_ensemble_multiaspect, load_benchmark, project_all_aspects

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
RESULTS_JSON = BASE_DIR / "data" / "train_seeds_results.json"
BEST_CONFIG_JSON = BASE_DIR / "data" / "best_ensemble_config.json"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

WEEK6_IMAGE_ENSEMBLE = {10: 0.1767, 30: 0.3054, 50: 0.3828}
PHASE28_TEXT_ENSEMBLE = {10: 0.1904, 30: 0.3267, 50: 0.4079}


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
    members = []
    for seed in seeds_used:
        r = by_seed[seed]
        ckpt = REPO_ROOT / r["local_checkpoint"]
        model, proj = project_all_aspects(ckpt, base_repr_t, DEVICE)
        members.append((model, proj))
        print(f"  seed={seed} projected")

    ensemble_recall, n, skip = evaluate_recall_ensemble_multiaspect(pools, queries, item_ids, members, DEVICE)
    print(f"Multi-aspect ensemble test recall: {ensemble_recall} (n={n}, skip={skip})")

    beats_text_ensemble = all(ensemble_recall[k] > PHASE28_TEXT_ENSEMBLE[k] for k in (10, 30, 50))
    gain_10 = ensemble_recall[10] / PHASE28_TEXT_ENSEMBLE[10] - 1

    lines = [
        "# Phase 30, Step 5: Final Evaluation -- Test Benchmark, Once\n",
        "Matched-conditions comparison against phase 28's text ensemble, our own reproduction of the "
        "CIR protocol, not a claim against OutfitTransformer's original test file.\n",
        f"Best ensemble composition identified entirely through validation-benchmark comparisons "
        f"(steps 3-4): size={best_cfg['best_size']}, seeds={seeds_used}, "
        f"validation Recall@10={best_cfg['val_recall10']:.4f}. Evaluated here, exactly once, on the "
        "actual test CIR benchmark.\n",
        "## Full progression\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Week 6 image-only, 10-model ensemble | {WEEK6_IMAGE_ENSEMBLE[10]:.4f} | {WEEK6_IMAGE_ENSEMBLE[30]:.4f} | {WEEK6_IMAGE_ENSEMBLE[50]:.4f} |",
        f"| Phase 28 text-only, 10-model ensemble | {PHASE28_TEXT_ENSEMBLE[10]:.4f} | {PHASE28_TEXT_ENSEMBLE[30]:.4f} | {PHASE28_TEXT_ENSEMBLE[50]:.4f} |",
        f"| **Phase 30 multi-aspect, {best_cfg['best_size']}-model ensemble (seeds {seeds_used})** | **{ensemble_recall[10]:.4f}** | **{ensemble_recall[30]:.4f}** | **{ensemble_recall[50]:.4f}** |",
        "",
    ]

    if beats_text_ensemble:
        lines.append(
            f"**The multi-aspect ensemble beats phase 28's text-only ensemble at every K.** "
            f"Relative gain at K=10: {gain_10:+.1%}."
        )
    else:
        lines.append(
            "**The multi-aspect ensemble does NOT beat phase 28's text-only ensemble cleanly at every K** "
            "-- reported honestly, see `phase30_notes.md` for the full interpretation."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

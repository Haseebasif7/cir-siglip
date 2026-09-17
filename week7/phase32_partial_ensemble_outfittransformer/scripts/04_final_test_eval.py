"""
Phase 32, step 5: the single final test-benchmark evaluation. Takes the
3-seed ensemble (42, 1, 2), identified and validated entirely through
validation-benchmark comparisons (step 4), and evaluates it exactly once on
the actual test CIR benchmark -- the single test-benchmark touch of this
entire phase.
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
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"  # the ONE test-benchmark touch of this phase
MODELS_DIR = BASE_DIR / "models"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEEDS = [42, 1, 2]
PHASE31_SINGLE = {10: 0.1799, 30: 0.3111, 50: 0.3844}
PHASE28_ENSEMBLE = {10: 0.1904, 30: 0.3267, 50: 0.4079}
PHASE14B_ORIGINAL = {10: 0.0588, 30: 0.1286, 50: 0.1809}


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

    pools, queries = load_benchmark(TEST_BENCHMARK)
    print(f"Loaded TEST benchmark: {len(queries)} queries -- the single touch of this phase.")

    print("Loading and projecting all 3 members...")
    members = []
    for seed in SEEDS:
        ckpt = MODELS_DIR / f"ot32_seed{seed}.pt"
        model = load_member(ckpt, DEVICE)
        cand_emb = precompute_candidates(model, base_repr, DEVICE)
        members.append((model, cand_emb))
        print(f"  seed={seed} projected")

    ens_recall, n_total, n_skipped = evaluate_recall_ensemble(pools, queries, idx, base_repr, members, DEVICE)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Phase 32, 3-seed ensemble, test benchmark: "
          f"R@10={ens_recall[10]:.4f} R@30={ens_recall[30]:.4f} R@50={ens_recall[50]:.4f}")

    beats_phase31_single = all(ens_recall[k] > PHASE31_SINGLE[k] for k in (10, 30, 50))
    beats_phase28 = all(ens_recall[k] > PHASE28_ENSEMBLE[k] for k in (10, 30, 50))
    pct_of_phase28 = {k: ens_recall[k] / PHASE28_ENSEMBLE[k] for k in (10, 30, 50)}
    gain_over_phase31 = {k: ens_recall[k] / PHASE31_SINGLE[k] - 1 for k in (10, 30, 50)}
    gain_over_14b = {k: ens_recall[k] / PHASE14B_ORIGINAL[k] - 1 for k in (10, 30, 50)}

    lines = [
        "# Phase 32, Step 5: Final Evaluation -- Test Benchmark, Once\n",
        f"3-seed ensemble (42, 1, 2), score-averaged, identified entirely through validation-benchmark "
        f"comparisons (step 4: val R@10=0.2035, +4.8% over the best solo seed). Evaluated here, exactly "
        f"once, on the actual test CIR benchmark.\n",
        "## Full progression\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |",
        "|---|---|---|---|---|",
        f"| Phase 14b original (single config, un-invested) | {PHASE14B_ORIGINAL[10]:.4f} | {PHASE14B_ORIGINAL[30]:.4f} | {PHASE14B_ORIGINAL[50]:.4f} | superseded baseline |",
        f"| Phase 31 single model (fully tuned, no ensemble) | {PHASE31_SINGLE[10]:.4f} | {PHASE31_SINGLE[30]:.4f} | {PHASE31_SINGLE[50]:.4f} | |",
        f"| **Phase 32, 3-seed partial ensemble** | **{ens_recall[10]:.4f}** | **{ens_recall[30]:.4f}** | **{ens_recall[50]:.4f}** | budget-constrained, 3 of a possible 10 seeds |",
        f"| Phase 28 text ensemble (project's own best, 10-model) | {PHASE28_ENSEMBLE[10]:.4f} | {PHASE28_ENSEMBLE[30]:.4f} | {PHASE28_ENSEMBLE[50]:.4f} | |",
        "",
        "## Derived comparisons\n",
        f"- vs. phase 31's single model: {gain_over_phase31[10]:+.1%} / {gain_over_phase31[30]:+.1%} / {gain_over_phase31[50]:+.1%} relative "
        f"({'beats' if beats_phase31_single else 'does NOT beat'} it at every K).",
        f"- vs. phase 14b's original: {gain_over_14b[10]:+.1%} / {gain_over_14b[30]:+.1%} / {gain_over_14b[50]:+.1%} relative.",
        f"- vs. phase 28's own 10-model ensemble: {pct_of_phase28[10]:.1%} / {pct_of_phase28[30]:.1%} / {pct_of_phase28[50]:.1%} "
        f"({'BEATS' if beats_phase28 else 'still trails'} it at every K).",
        "",
        f"n_total={n_total} n_skipped={n_skipped}.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")

    with open(BASE_DIR / "data" / "final_test_result.json", "w") as f:
        json.dump({"recall": ens_recall, "n_total": n_total, "n_skipped": n_skipped, "seeds": SEEDS}, f, indent=2)


if __name__ == "__main__":
    main()

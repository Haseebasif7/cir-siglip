"""
Phase 33, step 5: the single final test-benchmark evaluation. Takes the
3-seed ensemble, identified and validated entirely through validation-
benchmark comparisons (step 4), and evaluates it exactly once on the actual
test CIR benchmark -- the single test-benchmark touch of this entire phase.
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
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEG_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"  # the ONE test-benchmark touch of this phase
METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"
MODELS_DIR = BASE_DIR / "models"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEEDS = [42, 1, 2]
PHASE13B_ORIGINAL = {10: 0.0725, 30: 0.1393, 50: 0.1844}
PHASE28_ENSEMBLE = {10: 0.1904, 30: 0.3267, 50: 0.4079}
PHASE32_OT_PARITY = {10: 0.1897, 30: 0.3246, 50: 0.4019}


def main():
    print("Loading catalog...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA, NEG_CANDIDATES,
                                        text_embeddings_npz=TEXT_EMBEDDINGS_NPZ, metadata_path=METADATA_JSON,
                                        device=DEVICE)
    with open(TEST_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]
    print(f"Loaded TEST benchmark: {len(queries)} queries -- the single touch of this phase.")

    print("Loading all 3 members...")
    models = []
    for seed in SEEDS:
        m = CSANetSigLIP(num_categories=len(catalog["cat_to_idx"]), siglip_dim=catalog["in_dim"]).to(DEVICE).eval()
        m.load_state_dict(torch.load(MODELS_DIR / f"ot33_seed{seed}.pt", map_location=DEVICE))
        models.append(m)
        print(f"  seed={seed} loaded")

    ens_recall, n_total, n_skipped = train_core.evaluate_recall_ensemble(
        models, catalog["base_repr"], catalog["id_to_gidx"], catalog["cat_to_idx"], catalog["item_cat"],
        pools, queries, DEVICE,
    )
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Phase 33, 3-seed CSA-Net ensemble, test benchmark: "
          f"R@10={ens_recall[10]:.4f} R@30={ens_recall[30]:.4f} R@50={ens_recall[50]:.4f}")

    gain_over_13b = {k: ens_recall[k] / PHASE13B_ORIGINAL[k] - 1 for k in (10, 30, 50)}
    pct_of_phase28 = {k: ens_recall[k] / PHASE28_ENSEMBLE[k] for k in (10, 30, 50)}
    vs_phase32 = {k: ens_recall[k] / PHASE32_OT_PARITY[k] - 1 for k in (10, 30, 50)}
    beats_phase28 = all(ens_recall[k] > PHASE28_ENSEMBLE[k] for k in (10, 30, 50))
    beats_phase32 = all(ens_recall[k] > PHASE32_OT_PARITY[k] for k in (10, 30, 50))

    lines = [
        "# Phase 33, Step 5: Final Evaluation -- Test Benchmark, Once\n",
        f"3-seed ensemble (42, 1, 2), score-averaged (distances, not embeddings), identified entirely "
        f"through validation-benchmark comparisons (step 4: val R@10=0.1318, +22.7% over the best solo "
        f"seed). Evaluated here, exactly once, on the actual test CIR benchmark.\n",
        "## Full progression\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |",
        "|---|---|---|---|---|",
        f"| Phase 13b original CSA-Net reproduction (un-invested) | {PHASE13B_ORIGINAL[10]:.4f} | {PHASE13B_ORIGINAL[30]:.4f} | {PHASE13B_ORIGINAL[50]:.4f} | superseded baseline |",
        f"| **Phase 33, 3-seed investment-parity CSA-Net ensemble** | **{ens_recall[10]:.4f}** | **{ens_recall[30]:.4f}** | **{ens_recall[50]:.4f}** | this phase |",
        f"| Phase 32, investment-parity OutfitTransformer (3-seed) | {PHASE32_OT_PARITY[10]:.4f} | {PHASE32_OT_PARITY[30]:.4f} | {PHASE32_OT_PARITY[50]:.4f} | cross-architecture comparison |",
        f"| Phase 28 text ensemble (project's own best, 10-model) | {PHASE28_ENSEMBLE[10]:.4f} | {PHASE28_ENSEMBLE[30]:.4f} | {PHASE28_ENSEMBLE[50]:.4f} | |",
        "",
        "## Derived comparisons\n",
        f"- vs. phase 13b's original: {gain_over_13b[10]:+.1%} / {gain_over_13b[30]:+.1%} / {gain_over_13b[50]:+.1%} relative.",
        f"- vs. phase 28's own 10-model ensemble: {pct_of_phase28[10]:.1%} / {pct_of_phase28[30]:.1%} / {pct_of_phase28[50]:.1%} "
        f"({'BEATS' if beats_phase28 else 'trails'} it at every K).",
        f"- vs. phase 32's investment-parity OutfitTransformer (the direct cross-architecture comparison this phase exists to make): "
        f"{vs_phase32[10]:+.1%} / {vs_phase32[30]:+.1%} / {vs_phase32[50]:+.1%} relative "
        f"({'BEATS' if beats_phase32 else 'trails'} it at every K).",
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

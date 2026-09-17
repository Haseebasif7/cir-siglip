"""
Phase 23, step 7: the single final evaluation. The one best configuration
identified entirely on the validation benchmark (steps 3-6) is evaluated
here, exactly once, on the actual test CIR benchmark used throughout this
project (week4/phase12_controllable_modes/data/cir_benchmark.json) -- the
same benchmark, same evaluate_recall() protocol phase 9/20's own
verification used, so the improvement (if any) over phase 9's original
cited number is measured honestly and cleanly. This script must only be run
ONCE, after every tuning decision in steps 3-6 is already finalized.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead
from cir_eval import evaluate_recall, load_benchmark

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE9_CKPT = PHASE9_DIR / "models" / "model_a_random_negs.pt"
TUNED_CKPT = BASE_DIR / "models" / "final_tuned.pt"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
FINAL_CONFIG_JSON = BASE_DIR / "data" / "final_config_train_result.json"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PHASE9_ORIGINAL_RECALL = {10: 0.1317, 30: 0.2464, 50: 0.3216}


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    pools, queries = load_benchmark(TEST_BENCHMARK)

    def eval_ckpt(ckpt_path):
        model = ProjectionHead().to(DEVICE).eval()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        with torch.no_grad():
            proj = model(torch.tensor(embeddings, device=DEVICE)).cpu().numpy()
        return evaluate_recall(pools, queries, item_ids, proj)

    p9_recall, p9_n, p9_skip = eval_ckpt(PHASE9_CKPT)
    tuned_recall, tuned_n, tuned_skip = eval_ckpt(TUNED_CKPT)

    with open(FINAL_CONFIG_JSON) as f:
        final_train_result = json.load(f)
    final_cfg = final_train_result["config"]

    print(f"Phase 9 original: {p9_recall} (n={p9_n}, skip={p9_skip})")
    print(f"Phase 23 tuned:   {tuned_recall} (n={tuned_n}, skip={tuned_skip})")

    def pct_change(new, old):
        return 100.0 * (new - old) / old if old else float("nan")

    lines = [
        "# Phase 23, Step 7: Final Evaluation -- Test Benchmark, Once",
        "",
        "The single best configuration identified entirely through validation-benchmark "
        "comparisons (steps 3-6, `tuning_log.md`) evaluated here, exactly once, on the "
        "actual test CIR benchmark used throughout this project -- verifying that any "
        "improvement is real, not a validation-benchmark artifact.",
        "",
        "## Final tuned configuration\n",
        "| Hyperparameter | Phase 9 original | Phase 23 tuned |",
        "|---|---|---|",
        f"| Learning rate | 1e-3 | {final_cfg['lr']} |",
        f"| Batch size | 128 | {final_cfg['batch_size']} |",
        f"| Weight decay | 1e-5 | {final_cfg['weight_decay']} |",
        f"| Temperature (tau) | 0.07 | {final_cfg['tau']} |",
        f"| Random negatives (R) | 8 | {final_cfg['r_neg']} |",
        f"| Selection signal | validation loss (best at epoch 0) | validation-benchmark Recall@10 "
        f"(best at epoch {final_train_result['best_epoch']} of {final_train_result['n_epochs_run']} run) |",
        "",
        "## Test-benchmark result (the one and only comparison that matters)\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Phase 9 original (cited) | {PHASE9_ORIGINAL_RECALL[10]:.4f} | {PHASE9_ORIGINAL_RECALL[30]:.4f} | {PHASE9_ORIGINAL_RECALL[50]:.4f} |",
        f"| Phase 9 original (re-verified here, same checkpoint) | {p9_recall[10]:.4f} | {p9_recall[30]:.4f} | {p9_recall[50]:.4f} |",
        f"| **Phase 23 tuned** | **{tuned_recall[10]:.4f}** | **{tuned_recall[30]:.4f}** | **{tuned_recall[50]:.4f}** |",
        "",
        "## Change over phase 9's original\n",
        "| K | Absolute change | Relative change |",
        "|---|---|---|",
    ]
    for k in (10, 30, 50):
        abs_change = tuned_recall[k] - p9_recall[k]
        lines.append(f"| {k} | {abs_change:+.4f} | {pct_change(tuned_recall[k], p9_recall[k]):+.1f}% |")
    lines.append("")

    improved = all(tuned_recall[k] > p9_recall[k] for k in (10, 30, 50))
    if improved:
        lines.append(
            "**Tuning produced a real improvement over phase 9's original result, on the "
            "test benchmark, at every K.** This was verified with a single, final "
            "test-benchmark evaluation only after every tuning decision was already locked "
            "in on the separate validation benchmark -- not selected by repeatedly checking "
            "against this specific benchmark."
        )
    else:
        lines.append(
            "**Tuning did NOT produce a clean improvement across every K on the test "
            "benchmark**, despite validation-benchmark gains -- reported honestly, see "
            "`phase23_notes.md` for the full interpretation of what this means."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

"""
Phase 25, step 6: the single final evaluation. Only runs if step 5's
selection found a scaled configuration that beats phase 23/24's baseline on
the validation benchmark. Evaluates that one configuration, exactly once, on
the actual test CIR benchmark, reporting it directly alongside phase 9's
original number and phase 23/24's current best (0.1473/0.2684/0.3442).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE9_CKPT = PHASE9_DIR / "models" / "model_a_random_negs.pt"
TUNED_CKPT_128 = REPO_ROOT / "week4/phase23_hyperparameter_tuning/models/final_tuned.pt"
SCALED_CKPT = BASE_DIR / "models" / "final_scaled.pt"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
FINAL_CONFIG_JSON = BASE_DIR / "data" / "final_config_train_result.json"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PHASE9_ORIGINAL_RECALL = {10: 0.1317, 30: 0.2464, 50: 0.3216}
PHASE23_24_RECALL = {10: 0.1473, 30: 0.2684, 50: 0.3442}


class ProjectionHeadOriginal(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ProjectionHeadGeneral(nn.Module):
    def __init__(self, in_dim=768, hidden_dims=(256,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def main():
    with open(FINAL_CONFIG_JSON) as f:
        final_result = json.load(f)

    if final_result.get("outcome") != "scale_wins":
        print("Step 5 found the baseline (width=256/depth=1/out_dim=128) still wins -- "
              "no new checkpoint to evaluate. Skipping step 6, writing a short explanatory note.")
        OUT_MD.write_text(
            "# Phase 25, Step 6: Test-Benchmark Check -- Not Warranted\n\n"
            "No scaled configuration (width, depth, or embedding-dimension) beat phase 23/24's "
            "established baseline (hidden_dims=[256], out_dim=128, lr=0.001; validation "
            "Recall@10=0.1600) on the validation benchmark -- see `width_sweep.md`, "
            "`depth_sweep.md`, `embedding_dim_sweep.md` for the full per-configuration results. "
            "Per this phase's own brief (report a clear negative finding directly rather than "
            "evaluate a configuration the validation benchmark never actually selected), no new "
            "test-benchmark evaluation was run.\n\n"
            "**Phase 23/24's result stands as the current, final, honest number:**\n\n"
            "| Configuration | Recall@10 | Recall@30 | Recall@50 |\n"
            "|---|---|---|---|\n"
            f"| Phase 9 original | {PHASE9_ORIGINAL_RECALL[10]:.4f} | {PHASE9_ORIGINAL_RECALL[30]:.4f} | {PHASE9_ORIGINAL_RECALL[50]:.4f} |\n"
            f"| **Phase 23/24 tuned (current best, architecture unchanged from phase 9)** | "
            f"**{PHASE23_24_RECALL[10]:.4f}** | **{PHASE23_24_RECALL[30]:.4f}** | **{PHASE23_24_RECALL[50]:.4f}** |\n\n"
            "See `phase25_notes.md` for the full honest interpretation of why scale did not help here.\n"
        )
        print(f"Saved {OUT_MD}")
        return

    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    pools, queries = load_benchmark(TEST_BENCHMARK)

    def eval_ckpt(model, ckpt_path):
        model = model.to(DEVICE).eval()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        with torch.no_grad():
            proj = model(torch.tensor(embeddings, device=DEVICE)).cpu().numpy()
        return evaluate_recall(pools, queries, item_ids, proj)

    p9_recall, _, _ = eval_ckpt(ProjectionHeadOriginal(), PHASE9_CKPT)
    tuned128_recall, _, _ = eval_ckpt(ProjectionHeadOriginal(), TUNED_CKPT_128)

    cfg = final_result["config"]
    scaled_model = ProjectionHeadGeneral(hidden_dims=cfg["hidden_dims"], out_dim=cfg["out_dim"])
    scaled_recall, n, skip = eval_ckpt(scaled_model, SCALED_CKPT)

    print(f"Phase 9 original: {p9_recall}")
    print(f"Phase 23/24 tuned (128-d, original arch): {tuned128_recall}")
    print(f"Phase 25 scaled: {scaled_recall} (n={n}, skip={skip})")

    lines = [
        "# Phase 25, Step 6: Final Evaluation -- Test Benchmark, Once",
        "",
        "The single best scaled configuration identified entirely through validation-benchmark "
        "comparisons (steps 1-3) evaluated here, exactly once, on the actual test CIR benchmark.",
        "",
        "## Final scaled configuration\n",
        f"- hidden_dims: {cfg['hidden_dims']}",
        f"- out_dim: {cfg['out_dim']}",
        f"- lr: {cfg['lr']} (batch_size=256, weight_decay=0.0, tau=0.15, R=8 held fixed from phase 23/24)",
        f"- Parameters: {final_result.get('n_params', 'n/a')}",
        f"- Selected at epoch {final_result['best_epoch']} of {final_result['n_epochs_run']} run "
        f"(validation Recall@10={final_result['best_recall10']:.4f})",
        "",
        "## Test-benchmark result\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Phase 9 original | {p9_recall[10]:.4f} | {p9_recall[30]:.4f} | {p9_recall[50]:.4f} |",
        f"| Phase 23/24 tuned (128-d, original architecture) | {tuned128_recall[10]:.4f} | {tuned128_recall[30]:.4f} | {tuned128_recall[50]:.4f} |",
        f"| **Phase 25 scaled** | **{scaled_recall[10]:.4f}** | **{scaled_recall[30]:.4f}** | **{scaled_recall[50]:.4f}** |",
        "",
    ]

    improved = all(scaled_recall[k] > tuned128_recall[k] for k in (10, 30, 50))
    if improved:
        lines.append(
            "**Scale produced a real improvement over phase 23/24's tuned baseline, on the "
            "test benchmark, at every K.**"
        )
    else:
        lines.append(
            "**Scale did NOT produce a clean improvement across every K on the test benchmark, "
            "despite winning on the validation benchmark** -- reported honestly, see "
            "`phase25_notes.md` for the full interpretation."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

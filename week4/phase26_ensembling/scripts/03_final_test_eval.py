"""
Phase 26, step 6: the single final evaluation. Takes the best-performing
ensemble composition identified purely through validation-benchmark
comparisons (steps 3-5) and evaluates it exactly once on the actual test CIR
benchmark, reported directly alongside phase 25's single best model
(0.1505/0.2740/0.3503), phase 23/24's tuned single model (0.1473/0.2684/
0.3442), and phase 9's original number (0.1317/0.2464/0.3216).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval_ensemble import evaluate_recall_ensemble, load_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"

RESULTS_JSON = BASE_DIR / "data" / "train_seeds_results.json"
BEST_CONFIG_JSON = BASE_DIR / "data" / "best_ensemble_config.json"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PHASE9_ORIGINAL_RECALL = {10: 0.1317, 30: 0.2464, 50: 0.3216}
PHASE23_24_RECALL = {10: 0.1473, 30: 0.2684, 50: 0.3442}
PHASE25_SCALE_RECALL = {10: 0.1505, 30: 0.2740, 50: 0.3503}


class ProjectionHeadGeneral(nn.Module):
    def __init__(self, in_dim=768, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def project_all(ckpt_path, hidden_dims, out_dim, base_emb_t):
    model = ProjectionHeadGeneral(hidden_dims=hidden_dims, out_dim=out_dim).to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    with torch.no_grad():
        proj = model(base_emb_t).cpu().numpy()
    return proj


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    with open(BEST_CONFIG_JSON) as f:
        best_cfg = json.load(f)

    use_mixed = best_cfg.get("diversity_improved")
    if use_mixed:
        seeds_used = best_cfg["mixed_seeds_used"]
        val_r10 = best_cfg["mixed_val_recall10"]
        composition_desc = f"mixed: {seeds_used} (includes width=512 diversity member(s))"
    else:
        seeds_used = best_cfg["best_seeds"]
        val_r10 = best_cfg["val_recall10"]
        composition_desc = f"same-architecture (width=1024): seeds {seeds_used}"

    by_name = {r["config"]["name"]: r for r in results}
    main_by_seed = {r["config"]["seed"]: r for r in results if r["config"]["name"].startswith("ensemble_seed")
                     or r["config"]["name"] == "final_scaled"}
    div_by_seed = {r["config"]["seed"]: r for r in results if r["config"]["name"].startswith("diversity_width512")}

    members = []
    for seed in seeds_used:
        r = main_by_seed.get(seed) or div_by_seed.get(seed)
        members.append(r)

    print("Loading base SigLIP embeddings...")
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    base_emb = data["embeddings"].astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb, axis=1, keepdims=True)
    base_emb_t = torch.tensor(base_emb, device=DEVICE)

    pools, queries = load_benchmark(TEST_BENCHMARK)

    print(f"Projecting {len(members)} ensemble member(s) for the final test evaluation...")
    projections = []
    for r in members:
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, r["config"]["hidden_dims"], r["config"]["out_dim"], base_emb_t)
        projections.append(proj)
        print(f"  seed={r['config']['seed']} hidden_dims={r['config']['hidden_dims']} projected")

    ensemble_recall, n, skip = evaluate_recall_ensemble(pools, queries, item_ids, projections)
    print(f"Ensemble test recall: {ensemble_recall} (n={n}, skip={skip})")

    # also report the single best solo seed on the test benchmark for direct comparison
    seed42 = main_by_seed[42]
    ckpt42 = REPO_ROOT / seed42["local_checkpoint"]
    proj42 = project_all(ckpt42, seed42["config"]["hidden_dims"], seed42["config"]["out_dim"], base_emb_t)
    solo_recall, _, _ = evaluate_recall_ensemble(pools, queries, item_ids, [proj42])
    print(f"Solo (seed=42, same as phase 25's reported single-best) test recall: {solo_recall}")

    lines = [
        "# Phase 26, Step 6: Final Evaluation -- Test Benchmark, Once",
        "",
        f"Best ensemble composition identified entirely through validation-benchmark comparisons "
        f"(steps 3-5): {composition_desc}, validation Recall@10={val_r10:.4f}. Evaluated here, "
        "exactly once, on the actual test CIR benchmark.",
        "",
        "## Full progression\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Phase 9 original | {PHASE9_ORIGINAL_RECALL[10]:.4f} | {PHASE9_ORIGINAL_RECALL[30]:.4f} | {PHASE9_ORIGINAL_RECALL[50]:.4f} |",
        f"| Phase 23/24 tuned (single model) | {PHASE23_24_RECALL[10]:.4f} | {PHASE23_24_RECALL[30]:.4f} | {PHASE23_24_RECALL[50]:.4f} |",
        f"| Phase 25 scaled (single model, width=1024) | {PHASE25_SCALE_RECALL[10]:.4f} | {PHASE25_SCALE_RECALL[30]:.4f} | {PHASE25_SCALE_RECALL[50]:.4f} |",
        f"| Phase 26 solo (seed=42, single model, re-measured here for a like-for-like reference point) | {solo_recall[10]:.4f} | {solo_recall[30]:.4f} | {solo_recall[50]:.4f} |",
        f"| **Phase 26 ensemble ({composition_desc})** | **{ensemble_recall[10]:.4f}** | **{ensemble_recall[30]:.4f}** | **{ensemble_recall[50]:.4f}** |",
        "",
    ]

    improved = all(ensemble_recall[k] > PHASE25_SCALE_RECALL[k] for k in (10, 30, 50))
    if improved:
        lines.append(
            "**Ensembling produced a real improvement over phase 25's single best model, on the "
            "test benchmark, at every K.**"
        )
    else:
        lines.append(
            "**Ensembling did NOT produce a clean improvement across every K on the test benchmark "
            "over phase 25's single best model** -- reported honestly, see `phase26_notes.md` for "
            "the full interpretation."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

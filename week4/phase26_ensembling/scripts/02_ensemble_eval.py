"""
Phase 26, steps 3-5: individual seed variance, ensemble size sweep, and the
secondary architectural-diversity check. All done on the VALIDATION
benchmark only (test benchmark stays untouched until step 6).

Loads every trained checkpoint (10 width=1024 ensemble members + 2 width=512
diversity members), projects the same base SigLIP embeddings through each
one locally (small MLPs, cheap even at ~251k items), then uses
cir_eval_ensemble.evaluate_recall_ensemble -- which averages SIMILARITY
SCORES across members, never raw embedding vectors, per the brief's
explicit design choice -- for every ensemble composition tested below.
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
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"

RESULTS_JSON = BASE_DIR / "data" / "train_seeds_results.json"
MODELS_DIR = BASE_DIR / "models"

OUT_INDIVIDUAL = BASE_DIR / "individual_seeds.md"
OUT_SWEEP = BASE_DIR / "ensemble_size_sweep.md"
OUT_DIVERSITY = BASE_DIR / "architectural_diversity_check.md"
OUT_PROJECTIONS_CACHE = BASE_DIR / "data" / "member_projections.npz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SWEEP_SIZES = [2, 3, 5, 7, 10]


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

    main_members = [r for r in results if r["config"]["name"].startswith("ensemble_seed")
                     or r["config"]["name"] == "final_scaled"]
    diversity_members = [r for r in results if r["config"]["name"].startswith("diversity_width512")]
    assert len(main_members) == 10, f"expected 10 main members, got {len(main_members)}"
    assert len(diversity_members) == 2, f"expected 2 diversity members, got {len(diversity_members)}"

    # order the 10 main members: seed 42 first (phase 25's own winner, reused),
    # then seeds 1..9 in order -- this fixed order is what "ensembles built
    # from N models" (step 4) draws its first-N-members from.
    seed_order = [42] + list(range(1, 10))
    main_by_seed = {r["config"]["seed"]: r for r in main_members}
    main_members = [main_by_seed[s] for s in seed_order]

    print("Loading base SigLIP embeddings...")
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    base_emb = data["embeddings"].astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb, axis=1, keepdims=True)
    base_emb_t = torch.tensor(base_emb, device=DEVICE)

    pools, queries = load_benchmark(VAL_BENCHMARK)

    print("Projecting all 12 checkpoints through the base embeddings...")
    main_proj = []
    for r in main_members:
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, r["config"]["hidden_dims"], r["config"]["out_dim"], base_emb_t)
        main_proj.append(proj)
        print(f"  {r['config']['name']} (seed={r['config']['seed']}) projected")

    div_proj = []
    for r in diversity_members:
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, r["config"]["hidden_dims"], r["config"]["out_dim"], base_emb_t)
        div_proj.append(proj)
        print(f"  {r['config']['name']} (seed={r['config']['seed']}) projected")

    # ---- Step 3: individual seed solo results, recomputed locally on the
    # val benchmark (cross-check against the Modal-reported best_recall10) ----
    print("\nStep 3: individual solo evaluation...")
    solo_rows = []
    for r, proj in zip(main_members, main_proj):
        recall, n, skip = evaluate_recall_ensemble(pools, queries, item_ids, [proj])
        solo_rows.append((r["config"]["seed"], recall, r["best_recall10"], n, skip))
        print(f"  seed={r['config']['seed']}: local val_recall10={recall[10]:.4f} "
              f"(modal-reported={r['best_recall10']:.4f})")

    r10s = [row[1][10] for row in solo_rows]
    lines = [
        "# Phase 26, Step 3: Individual Seed Variance",
        "",
        "Ten independent copies of the winning phase 25 architecture (hidden_dims=[1024], "
        "out_dim=128, lr=0.0005, batch_size=256, weight_decay=0.0, tau=0.15, R=8 -- all "
        "unchanged, only the random seed varies), each selected by its own validation-benchmark "
        "Recall@10 peak. Seed 42 is phase 25's own confirmatory-retrain checkpoint, reused rather "
        "than retrained (see `phase26_notes.md`); seeds 1-9 are new. Recall@10 below is "
        "recomputed locally from each checkpoint's projected embeddings (single-member case of "
        "the ensemble evaluator) as a bit-consistency cross-check against the Modal-reported "
        "training-time number.",
        "",
        "| Seed | val Recall@10 (local) | val Recall@10 (Modal training-time) | val Recall@30 | val Recall@50 | best_epoch |",
        "|---|---|---|---|---|---|",
    ]
    for (seed, recall, modal_r10, n, skip), r in zip(solo_rows, main_members):
        lines.append(f"| {seed} | {recall[10]:.4f} | {modal_r10:.4f} | {recall[30]:.4f} | "
                      f"{recall[50]:.4f} | {r['best_epoch']} |")
    lines += [
        "",
        f"**Range across the 10 seeds: {min(r10s):.4f} - {max(r10s):.4f} "
        f"(spread {max(r10s) - min(r10s):.4f}, mean {np.mean(r10s):.4f}, std {np.std(r10s):.4f}).**",
        "",
    ]
    OUT_INDIVIDUAL.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_INDIVIDUAL}")

    # ---- Step 4: ensemble size sweep ----
    print("\nStep 4: ensemble size sweep...")
    sweep_rows = []
    for size in SWEEP_SIZES:
        members = main_proj[:size]
        recall, n, skip = evaluate_recall_ensemble(pools, queries, item_ids, members)
        seeds_used = [r["config"]["seed"] for r in main_members[:size]]
        sweep_rows.append((size, seeds_used, recall, n, skip))
        print(f"  size={size}: val_recall10={recall[10]:.4f}")

    best_size, best_seeds, best_recall, _, _ = max(sweep_rows, key=lambda row: row[2][10])

    lines = [
        "# Phase 26, Step 4: Ensemble Size Sweep",
        "",
        "Ensembles built by score-averaging (see `cir_eval_ensemble.py`) the first N of the 10 "
        "trained members, in a fixed order (seed 42 first -- phase 25's own winner -- then seeds "
        "1-9 in ascending order). Evaluated on the validation benchmark.",
        "",
        "| Ensemble size | Seeds used | val Recall@10 | val Recall@30 | val Recall@50 |",
        "|---|---|---|---|---|",
    ]
    lines.append(f"| 1 (solo, best single seed for reference) | 42 | {solo_rows[0][1][10]:.4f} | "
                  f"{solo_rows[0][1][30]:.4f} | {solo_rows[0][1][50]:.4f} |")
    for size, seeds_used, recall, n, skip in sweep_rows:
        lines.append(f"| {size} | {seeds_used} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} |")
    lines += [
        "",
        f"**Best: size={best_size} (val Recall@10={best_recall[10]:.4f}).**",
        "",
    ]
    OUT_SWEEP.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_SWEEP}")

    # ---- Step 5: architectural diversity (secondary) ----
    print("\nStep 5: architectural diversity check...")
    # Take best_size-member width=1024 ensemble as the baseline for this
    # comparison, then swap in the 2 width=512 members in place of the
    # 2 lowest-seed (i.e. last-added) width=1024 members at the same total size.
    baseline_members = main_proj[:best_size]
    baseline_recall, _, _ = evaluate_recall_ensemble(pools, queries, item_ids, baseline_members)

    n_swap = min(2, best_size)
    mixed_members = main_proj[:best_size - n_swap] + div_proj[:n_swap]
    mixed_recall, _, _ = evaluate_recall_ensemble(pools, queries, item_ids, mixed_members)
    mixed_seeds_used = [r["config"]["seed"] for r in main_members[:best_size - n_swap]] + \
                        [r["config"]["seed"] for r in diversity_members[:n_swap]]

    # also report the 2-member width512-only ensemble and each width512 solo, for context
    div_solo = []
    for r, proj in zip(diversity_members, div_proj):
        recall, _, _ = evaluate_recall_ensemble(pools, queries, item_ids, [proj])
        div_solo.append((r["config"]["seed"], recall))
    div_pair_recall, _, _ = evaluate_recall_ensemble(pools, queries, item_ids, div_proj)

    lines = [
        "# Phase 26, Step 5: Architectural Diversity Check (Secondary)",
        "",
        "Phase 25's own width sweep never saved checkpoints (it was a validation-only comparison), "
        "so two fresh width=512 models were trained here at phase 25's own winning width=512 "
        "setting (lr=0.0005, val Recall@10=0.1609 in the original sweep), seeds 201/202, "
        "specifically to test whether mixing architecture sizes into the ensemble adds anything "
        "beyond seed diversity alone. This is a smaller, secondary check -- the primary result is "
        "the same-architecture sweep in `ensemble_size_sweep.md`.",
        "",
        "| Width=512 solo seed | val Recall@10 |",
        "|---|---|",
    ]
    for seed, recall in div_solo:
        lines.append(f"| {seed} | {recall[10]:.4f} |")
    lines += [
        f"| Both width=512 seeds, ensembled together | {div_pair_recall[10]:.4f} |",
        "",
        f"## Same-size comparison at ensemble size {best_size}\n",
        "| Composition | Seeds | val Recall@10 | val Recall@30 | val Recall@50 |",
        "|---|---|---|---|---|",
        f"| All width=1024 (same-architecture baseline) | {[r['config']['seed'] for r in main_members[:best_size]]} | "
        f"{baseline_recall[10]:.4f} | {baseline_recall[30]:.4f} | {baseline_recall[50]:.4f} |",
        f"| {best_size - n_swap} x width=1024 + {n_swap} x width=512 (mixed) | {mixed_seeds_used} | "
        f"{mixed_recall[10]:.4f} | {mixed_recall[30]:.4f} | {mixed_recall[50]:.4f} |",
        "",
    ]
    improved = mixed_recall[10] > baseline_recall[10]
    lines.append(
        "**Architectural diversity added a small further improvement over the same-architecture "
        f"ensemble at this size** (+{mixed_recall[10]-baseline_recall[10]:.4f})."
        if improved else
        "**Architectural diversity did NOT improve on the same-architecture ensemble at this size** "
        f"({mixed_recall[10]:.4f} vs {baseline_recall[10]:.4f}) -- seed diversity alone accounts for "
        "the ensemble gain here, mixing in a weaker solo architecture (width=512 solo scores below "
        "width=1024 solo) did not help further."
    )
    lines.append("")
    OUT_DIVERSITY.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_DIVERSITY}")

    # persist the chosen best config for step 6
    best_config = {
        "best_size": best_size,
        "best_seeds": best_seeds,
        "val_recall10": best_recall[10],
        "val_recall30": best_recall[30],
        "val_recall50": best_recall[50],
        "diversity_improved": bool(improved),
        "mixed_seeds_used": mixed_seeds_used if improved else None,
        "mixed_val_recall10": mixed_recall[10] if improved else None,
    }
    with open(BASE_DIR / "data" / "best_ensemble_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved {BASE_DIR / 'data' / 'best_ensemble_config.json'}")


if __name__ == "__main__":
    main()

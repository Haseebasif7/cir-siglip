"""
Phase 23, step 4: using the winning batch size and learning rate from step 3,
sweep the contrastive loss's temperature (TAU) -- a well-known sensitive
hyperparameter in this loss family that phase 9 never explicitly tuned
(inherited unchanged from phase 7/8's original choice of 0.07). Selected
using validation-benchmark Recall@10, same periodic-eval/best-by-recall
protocol as step 3.

WINNING_LR / WINNING_BS are filled in from 02_lr_bs_grid.py's own result
(data/lr_bs_grid_results.json) once that grid finishes -- not hardcoded in
advance, to avoid quietly assuming an outcome before it's measured.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
GRID_RESULTS = BASE_DIR / "data" / "lr_bs_grid_results.json"
OUT_JSON = BASE_DIR / "data" / "temperature_sweep_results.json"

TAU_GRID = [0.03, 0.05, 0.07, 0.10, 0.15]
MAX_EPOCHS = 12
PATIENCE = 4


def load_winning_lr_bs():
    with open(GRID_RESULTS) as f:
        results = json.load(f)
    best = max(results, key=lambda r: r["best_recall10"])
    return best["config"]["lr"], best["config"]["batch_size"], best["best_recall10"]


def main():
    winning_lr, winning_bs, winning_recall10 = load_winning_lr_bs()
    print(f"Step 3 winner: lr={winning_lr} bs={winning_bs} (val Recall@10={winning_recall10:.4f})")

    train_one = modal.Function.from_name("phase23-hp-tuning", "train_one")

    configs = []
    for tau in TAU_GRID:
        configs.append({
            "name": f"tau{tau}_lr{winning_lr}_bs{winning_bs}",
            "lr": winning_lr,
            "batch_size": winning_bs,
            "weight_decay": 1e-5,
            "tau": tau,
            "r_neg": 8,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": 1,
        })

    print(f"Launching {len(configs)} temperature configs in parallel on Modal...")
    results = list(train_one.map(configs))

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"tau={c['tau']}: best_recall10={r['best_recall10']:.4f} @epoch{r['best_epoch']} "
              f"({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()

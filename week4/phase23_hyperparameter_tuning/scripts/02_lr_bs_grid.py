"""
Phase 23, step 3: tune learning rate and batch size together (they interact
strongly in contrastive learning -- batch size controls how many in-batch
negatives each step sees), holding everything else at phase 9's original
values (WEIGHT_DECAY=1e-5, TAU=0.07, R=8/H=0). Selected using
validation-benchmark Recall@10, checked every epoch during training (not
just the final epoch) -- see modal_app.py's `train_one` for the periodic
eval + best-by-recall model selection/early-stopping logic.

A smoke test (2 epochs, phase 9's original LR/BS) measured ~112s/epoch on a
Modal T4 -- 12 epochs/config, run in parallel across the grid via
`train_one.map()`, is expected to take on the order of ~20-25 minutes
wall-clock for the whole grid, not 12x that sequentially.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_JSON = BASE_DIR / "data" / "lr_bs_grid_results.json"

LR_GRID = [5e-4, 1e-3, 2e-3]
BS_GRID = [64, 128, 256]
MAX_EPOCHS = 12
PATIENCE = 4


def main():
    train_one = modal.Function.from_name("phase23-hp-tuning", "train_one")

    configs = []
    for lr in LR_GRID:
        for bs in BS_GRID:
            configs.append({
                "name": f"lr{lr}_bs{bs}",
                "lr": lr,
                "batch_size": bs,
                "weight_decay": 1e-5,
                "tau": 0.07,
                "r_neg": 8,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "eval_every": 1,
            })

    print(f"Launching {len(configs)} configs in parallel on Modal...")
    results = list(train_one.map(configs))

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"lr={c['lr']} bs={c['batch_size']}: best_recall10={r['best_recall10']:.4f} "
              f"@epoch{r['best_epoch']} ({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()

"""
Phase 25, step 1: width. Holding depth (1 hidden layer) and final embedding
size (128) fixed, test wider hidden layers -- doubling (512) and
quadrupling (1024) the current 256-dimensional hidden layer -- with a small
learning-rate recheck at each width (per the brief: don't assume the tuned
lr carries over unchanged when only capacity changes). batch_size=256,
weight_decay=0.0, tau=0.15, r_neg=8 stay fixed at phase 23/24's tuned
values throughout (not re-opened here).

width=256 itself is not re-run -- it's phase 23/24's already-established,
multiply-reproduced baseline (val Recall@10=0.1600 at lr=0.001, bit-exact
across three independent runs); re-running it here would just reproduce
that same number a fourth time.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_JSON = BASE_DIR / "data" / "width_sweep_results.json"

WIDTHS = [512, 1024]
LR_GRID = [0.0005, 0.001, 0.002]
MAX_EPOCHS = 12
PATIENCE = 4


def main():
    train_one = modal.Function.from_name("phase25-scale", "train_one")

    configs = []
    for width in WIDTHS:
        for lr in LR_GRID:
            configs.append({
                "name": f"width{width}_lr{lr}",
                "hidden_dims": [width],
                "out_dim": 128,
                "lr": lr,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "eval_every": 1,
            })

    print(f"Launching {len(configs)} width x lr configs in parallel on Modal...")
    results = list(train_one.map(configs))

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"width={c['hidden_dims'][0]} lr={c['lr']}: best_recall10={r['best_recall10']:.4f} "
              f"@epoch{r['best_epoch']} ({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s, "
              f"{r['n_params']} params)")


if __name__ == "__main__":
    main()

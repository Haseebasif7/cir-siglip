"""
Phase 25, step 2: depth. Using the winning width from step 1, test adding
one or two additional hidden layers (2 and 3 hidden layers total, vs the
original/step-1 architectures' 1), each new hidden layer at the winning
width, with a small learning-rate recheck at each depth. out_dim stays 128;
batch_size/weight_decay/tau/r_neg stay fixed at phase 23/24's tuned values.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
WIDTH_RESULTS = BASE_DIR / "data" / "width_sweep_results.json"
OUT_JSON = BASE_DIR / "data" / "depth_sweep_results.json"

LR_GRID = [0.0005, 0.001, 0.002]
MAX_EPOCHS = 12
PATIENCE = 4


def load_winning_width():
    with open(WIDTH_RESULTS) as f:
        results = json.load(f)
    best = max(results, key=lambda r: r["best_recall10"])
    return best["config"]["hidden_dims"][0], best["best_recall10"]


def main():
    width, width_recall10 = load_winning_width()
    print(f"Step 1 winner: width={width} (val Recall@10={width_recall10:.4f})")

    train_one = modal.Function.from_name("phase25-scale", "train_one")

    configs = []
    for n_extra_layers in (1, 2):
        hidden_dims = [width] * (1 + n_extra_layers)
        for lr in LR_GRID:
            configs.append({
                "name": f"depth{1+n_extra_layers}_w{width}_lr{lr}",
                "hidden_dims": hidden_dims,
                "out_dim": 128,
                "lr": lr,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "eval_every": 1,
            })

    print(f"Launching {len(configs)} depth x lr configs in parallel on Modal...")
    results = list(train_one.map(configs))

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"depth={len(c['hidden_dims'])} width={c['hidden_dims'][0]} lr={c['lr']}: "
              f"best_recall10={r['best_recall10']:.4f} @epoch{r['best_epoch']} "
              f"({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s, {r['n_params']} params)")


if __name__ == "__main__":
    main()

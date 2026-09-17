"""
Phase 25, step 3: final embedding dimension. Using the winning width and
depth from steps 1-2, test out_dim in {256, 512} instead of 128 -- this
changes the actual space similarity is computed in, tested as its own
variable, with the same small learning-rate recheck discipline as steps 1-2.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
WIDTH_RESULTS = BASE_DIR / "data" / "width_sweep_results.json"
DEPTH_RESULTS = BASE_DIR / "data" / "depth_sweep_results.json"
OUT_JSON = BASE_DIR / "data" / "embedding_dim_sweep_results.json"

OUT_DIMS = [256, 512]
LR_GRID = [0.0005, 0.001, 0.002]
MAX_EPOCHS = 12
PATIENCE = 4


def load_winning_arch():
    with open(WIDTH_RESULTS) as f:
        width_results = json.load(f)
    best_width = max(width_results, key=lambda r: r["best_recall10"])

    with open(DEPTH_RESULTS) as f:
        depth_results = json.load(f)
    best_depth = max(depth_results, key=lambda r: r["best_recall10"])

    # Winning architecture overall is whichever of (step1's best width, 1
    # layer) or (step2's best depth config) scored higher -- depth sweep
    # only tested 2/3-layer variants of the winning width, so this
    # comparison covers the full step1+2 search space honestly.
    if best_depth["best_recall10"] > best_width["best_recall10"]:
        return best_depth["config"]["hidden_dims"], best_depth["best_recall10"], "step2 (depth)"
    return best_width["config"]["hidden_dims"], best_width["best_recall10"], "step1 (width)"


def main():
    hidden_dims, base_recall10, source = load_winning_arch()
    print(f"Winning architecture so far: hidden_dims={hidden_dims} (from {source}, "
          f"val Recall@10={base_recall10:.4f})")

    train_one = modal.Function.from_name("phase25-scale", "train_one")

    configs = []
    for out_dim in OUT_DIMS:
        for lr in LR_GRID:
            configs.append({
                "name": f"outdim{out_dim}_hd{'-'.join(map(str,hidden_dims))}_lr{lr}",
                "hidden_dims": hidden_dims,
                "out_dim": out_dim,
                "lr": lr,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "eval_every": 1,
            })

    print(f"Launching {len(configs)} out_dim x lr configs in parallel on Modal...")
    results = list(train_one.map(configs))

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"out_dim={c['out_dim']} lr={c['lr']}: best_recall10={r['best_recall10']:.4f} "
              f"@epoch{r['best_epoch']} ({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s, "
              f"{r['n_params']} params)")


if __name__ == "__main__":
    main()

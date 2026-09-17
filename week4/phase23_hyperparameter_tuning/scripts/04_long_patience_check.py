"""
Phase 23, step 5: using the best configuration found so far (winning
lr/batch_size from step 3, winning tau from step 4), retrain with a
meaningfully longer patience and epoch budget to check whether real
validation-benchmark Recall@K was still improving when phase 9's original
run stopped (it stopped at epoch 5, selected by validation LOSS which was
best at epoch 0 -- see phase23_notes.md), or whether it had genuinely
plateaued. This directly tests whether the original result was ever given a
fair chance to converge.

MAX_EPOCHS/PATIENCE here are deliberately much larger than every prior stage
in this phase (which used a coarse 12-epoch/patience-4 budget to make the
grid searches tractable) -- this is the one run in this phase built
specifically to answer the "did it plateau or was it cut off" question, so
it needs the room to actually plateau if it's going to.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
GRID_RESULTS = BASE_DIR / "data" / "lr_bs_grid_results.json"
TAU_RESULTS = BASE_DIR / "data" / "temperature_sweep_results.json"
OUT_JSON = BASE_DIR / "data" / "long_patience_result.json"

MAX_EPOCHS = 60
PATIENCE = 15


def load_best_config():
    with open(GRID_RESULTS) as f:
        grid = json.load(f)
    best_grid = max(grid, key=lambda r: r["best_recall10"])
    lr, bs = best_grid["config"]["lr"], best_grid["config"]["batch_size"]

    with open(TAU_RESULTS) as f:
        taus = json.load(f)
    best_tau_run = max(taus, key=lambda r: r["best_recall10"])
    tau = best_tau_run["config"]["tau"]
    return lr, bs, tau


def main():
    lr, bs, tau = load_best_config()
    print(f"Best config so far: lr={lr} bs={bs} tau={tau}. "
          f"Retraining with max_epochs={MAX_EPOCHS}, patience={PATIENCE}.")

    train_one = modal.Function.from_name("phase23-hp-tuning", "train_one")

    config = {
        "name": f"long_lr{lr}_bs{bs}_tau{tau}",
        "lr": lr,
        "batch_size": bs,
        "weight_decay": 1e-5,
        "tau": tau,
        "r_neg": 8,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "eval_every": 1,
        "save_checkpoint": True,
    }

    result = train_one.remote(config)

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT_JSON}")
    print(f"Best epoch: {result['best_epoch']} / {result['n_epochs_run']} run, "
          f"best_recall10={result['best_recall10']:.4f}, wall_time={result['wall_time_sec']:.0f}s")
    for pt in result["curve"]:
        print(pt)


if __name__ == "__main__":
    main()

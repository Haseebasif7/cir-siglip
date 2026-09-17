"""
Phase 23, step 6: lighter refinement pass -- a smaller check of weight decay
and the number of explicit random negatives per anchor (R), using the best
configuration established through steps 3-5 (lr/batch_size/tau, and whatever
epoch budget step 5 showed was actually needed) as the base. Deliberately a
smaller grid than step 3 ("given time allows", per the brief) -- weight decay
and R are secondary knobs compared to lr/batch_size/tau, so this stays a
light pass, not another full grid.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
GRID_RESULTS = BASE_DIR / "data" / "lr_bs_grid_results.json"
TAU_RESULTS = BASE_DIR / "data" / "temperature_sweep_results.json"
LONG_RESULT = BASE_DIR / "data" / "long_patience_result.json"
OUT_JSON = BASE_DIR / "data" / "refinement_results.json"

WD_GRID = [0.0, 1e-5, 1e-4]
R_GRID = [4, 8, 12]
PATIENCE = 6


def load_best_config():
    with open(GRID_RESULTS) as f:
        grid = json.load(f)
    best_grid = max(grid, key=lambda r: r["best_recall10"])
    lr, bs = best_grid["config"]["lr"], best_grid["config"]["batch_size"]

    with open(TAU_RESULTS) as f:
        taus = json.load(f)
    tau = max(taus, key=lambda r: r["best_recall10"])["config"]["tau"]

    with open(LONG_RESULT) as f:
        long_result = json.load(f)
    # use step 5's best_epoch + a margin as this stage's epoch budget, so
    # each refinement run gets roughly as much room to converge as step 5
    # showed was actually needed, without re-running step 5's full 60-epoch
    # budget nine more times.
    max_epochs = max(long_result["best_epoch"] + PATIENCE + 5, 15)
    return lr, bs, tau, max_epochs


def main():
    lr, bs, tau, max_epochs = load_best_config()
    print(f"Base config: lr={lr} bs={bs} tau={tau}. Refinement max_epochs={max_epochs}, patience={PATIENCE}.")

    train_one = modal.Function.from_name("phase23-hp-tuning", "train_one")

    configs = []
    for wd in WD_GRID:
        for r_neg in R_GRID:
            configs.append({
                "name": f"wd{wd}_r{r_neg}_lr{lr}_bs{bs}_tau{tau}",
                "lr": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "tau": tau,
                "r_neg": r_neg,
                "max_epochs": max_epochs,
                "patience": PATIENCE,
                "eval_every": 1,
            })

    print(f"Launching {len(configs)} refinement configs in parallel on Modal...")
    results = list(train_one.map(configs))

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"wd={c['weight_decay']} r_neg={c['r_neg']}: best_recall10={r['best_recall10']:.4f} "
              f"@epoch{r['best_epoch']} ({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()

"""
Phase 31, step 3c: budget check. Step 3's winner (lr=1.5e-4, bs=384,
uniformity_weight=0.1, margin=0.2, val_recall10=0.1864 at the 60-epoch
sweep budget) consistently hit best_epoch in the mid-50s across the loss-
shape sweep -- close to the 60-epoch cap, suggesting the sweep budget may
have cut off real, still-in-progress improvement (the direct analogue of
phase 23's own long-patience check). Confirms/reconciles at the full
100-epoch, patience=25 budget that worked cleanly for A2/A3-corrected.
Whichever budget's result is used becomes the standing config for step 4.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "budget_check_results.json"

WINNER_CONFIG = {
    "name": "ot31_budget_check_full", "input_mode": "image_text", "selection_metric": "recall10",
    "lr": 1.5e-4, "batch_size": 384, "uniformity_weight": 0.1, "margin": 0.2,
    "max_epochs": 100, "patience": 25, "save_checkpoint": True,
}


def main():
    results = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            results = {r["config"]["name"]: r for r in json.load(f)}

    if WINNER_CONFIG["name"] not in results:
        train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")
        result = train_one.remote(WINNER_CONFIG)
        print(f"{WINNER_CONFIG['name']}: val_recall10={result['best_recall10']:.4f} "
              f"best_epoch={result['best_epoch']} n_epochs_run={result['n_epochs_run']} "
              f"wall_time={result['wall_time_sec']:.0f}s")
        results[WINNER_CONFIG["name"]] = result
        with open(OUT_JSON, "w") as f:
            json.dump(list(results.values()), f, indent=2)
    else:
        print("Already done:", results[WINNER_CONFIG["name"]]["best_recall10"])

    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

"""
Phase 33, step 3 bracketing extension: the grid's winner (lr=1e-4, bs=48,
val_recall10=0.1075) sits at BS_GRID's bottom edge -- extend downward with
bs=24 before treating 48 as final, per the bracketing rule (phase 23/31's
own precedent).
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "tuning_grid_results.json"

CONFIGS = [
    {"name": "ot33_grid_lr1e-04_bs24", "input_mode": "image_text", "selection_metric": "recall10",
     "lr": 1e-4, "batch_size": 24, "patience": 5, "max_epochs": 40, "save_checkpoint": False},
]


def main():
    with open(OUT_JSON) as f:
        results = {r["config"]["name"]: r for r in json.load(f)}
    pending = [c for c in CONFIGS if c["name"] not in results]
    if pending:
        train_one = modal.Function.from_name("phase33-csanet-parity", "train_one")
        for cfg, result in zip(pending, train_one.map(pending)):
            print(f"{cfg['name']}: val_recall10={result['best_recall10']:.4f} "
                  f"best_epoch={result['best_epoch']} n_epochs_run={result['n_epochs_run']}")
            results[cfg["name"]] = result
    with open(OUT_JSON, "w") as f:
        json.dump(list(results.values()), f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

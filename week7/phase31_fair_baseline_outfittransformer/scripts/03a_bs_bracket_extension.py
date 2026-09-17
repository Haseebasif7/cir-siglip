"""
Phase 31, step 3a bracketing extension: the LR x BS grid's winner
(lr=1.5e-4, bs=384, val_recall10=0.0757) sits at BS_GRID's top edge (384 was
the largest batch size tried) -- per the plan's own bracketing rule ("if the
winner lands on a grid edge, run one extension round in that direction
before proceeding," phase 23's own tau-sweep lesson), extend upward with
bs=768 before treating 384 as final. Also note best_epoch=54 of a 60-epoch
budget for the winner -- close to the cap, addressed separately in
03c_budget_check.py rather than here (this script only resolves the grid
EDGE question, not the epoch-budget question).
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "lr_bs_grid_results.json"  # append into the same file

CONFIGS = [
    {"name": "ot31_grid_lr1.5e-4_bs768", "input_mode": "image_text", "selection_metric": "recall10",
     "lr": 1.5e-4, "batch_size": 768, "max_epochs": 60, "patience": 20, "save_checkpoint": False},
]


def main():
    with open(OUT_JSON) as f:
        results = {r["config"]["name"]: r for r in json.load(f)}
    pending = [c for c in CONFIGS if c["name"] not in results]
    if pending:
        train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")
        for cfg, result in zip(pending, train_one.map(pending)):
            print(f"{cfg['name']}: val_recall10={result['best_recall10']:.4f} "
                  f"best_epoch={result['best_epoch']} n_epochs_run={result['n_epochs_run']}")
            results[cfg["name"]] = result
    with open(OUT_JSON, "w") as f:
        json.dump(list(results.values()), f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

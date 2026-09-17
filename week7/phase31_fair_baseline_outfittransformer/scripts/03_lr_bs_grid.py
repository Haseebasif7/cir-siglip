"""
Phase 31, step 3a: joint LR x batch-size grid, mirroring phase 23's
discipline (selection by val Recall@10, every config reported, winner
bolded). Base configuration: input_mode="image_text" (step 1's winner,
architecture held fixed per the brief -- "hold the architecture itself
fixed at this stage" means the projection/transformer shape, not that text
gets reverted).

Sweep budget: max_epochs=60, patience=20 -- NOT phase 23's original 40/8.
Step 2's diagnostic (checkpoint_selection_check.md) found Recall@10 has a
genuine ~17-epoch plateau early in training on this architecture; patience=8
would very likely re-trigger the same premature stop across most of this
grid, making the comparison meaningless. patience=20 gives comfortable
margin above the observed 17-epoch plateau; max_epochs=60 gives ~40 epochs
of headroom after the plateau clears (A2/A3-corrected's own 100-epoch runs
show the climb continuing steadily well past epoch 60, so this sweep budget
is a deliberately cheaper proxy for RANKING configs relatively, not a claim
that 60 epochs reaches full convergence -- see step 3c's budget check for
that reconciliation).

LR_GRID centered wider/upward from the current lr=2e-5, not symmetrically
(see architecture_notes.md): phase 14b's original 92-epoch run never
resolved its triplet margin at any epoch -- under-training, not
over-stepping, is the diagnosed failure mode, and 2e-5 was inherited from a
reference repo that fine-tunes a full backbone, not a small frozen-backbone
head.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "lr_bs_grid_results.json"

LR_GRID = [1e-5, 2e-5, 5e-5, 1.5e-4, 5e-4]
BS_GRID = [96, 192, 384]
SWEEP_BUDGET = {"max_epochs": 60, "patience": 20}


def build_configs():
    configs = []
    for lr in LR_GRID:
        for bs in BS_GRID:
            name = f"ot31_grid_lr{lr:.0e}_bs{bs}".replace("+", "").replace("-0", "-")
            configs.append({
                "name": name, "input_mode": "image_text", "selection_metric": "recall10",
                "lr": lr, "batch_size": bs, "save_checkpoint": False,
                **SWEEP_BUDGET,
            })
    return configs


def load_existing_results():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save_results(results_by_name):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results_by_name.values()), f, indent=2)


def main():
    configs = build_configs()
    results_by_name = load_existing_results()
    pending = [c for c in configs if c["name"] not in results_by_name]

    if pending:
        train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")
        print(f"Launching {len(pending)} configs via .map()...")
        try:
            for cfg, result in zip(pending, train_one.map(pending)):
                print(f"{cfg['name']}: lr={cfg['lr']:.0e} bs={cfg['batch_size']} "
                      f"val_recall10={result['best_recall10']:.4f} best_epoch={result['best_epoch']} "
                      f"n_epochs_run={result['n_epochs_run']} wall_time={result['wall_time_sec']:.0f}s")
                results_by_name[cfg["name"]] = result
                save_results(results_by_name)
        except Exception as e:
            print(f"  .map() raised: {e!r} -- re-run this script; finished configs are saved.")

    print(f"\n{len(results_by_name)}/{len(configs)} configs done. Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

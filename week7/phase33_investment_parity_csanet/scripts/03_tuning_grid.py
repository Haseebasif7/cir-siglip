"""
Phase 33, step 3: joint LR x batch-size grid, mirroring phase 31's
discipline (selection by val Recall@10, every config reported, winner
bolded). Base configuration: input_mode=image_text (step 2's winner),
selection_metric=recall10, patience=5 (step 1's finding: unlike
OutfitTransformer, no recalibration needed for this architecture).

The smoke test measured ~9.2s/epoch on T4 including eval -- a full 40-epoch
run costs roughly $0.06-0.08, so (unlike phase 31's OutfitTransformer grid,
which needed a deliberately cheaper reduced-epoch sweep budget to stay
affordable) this phase can afford the FULL max_epochs=40 budget for every
grid point directly, no separate cheap-sweep-then-confirm step needed.

LR_GRID centered on phase 13b's own lr=5e-5 (not deliberately skewed the
way phase 31's OutfitTransformer grid was, since step 1 found no evidence
CSA-Net's original LR was mis-set the way OutfitTransformer's was).
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "tuning_grid_results.json"

LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
BS_GRID = [48, 96, 192]


def build_configs():
    configs = []
    for lr in LR_GRID:
        for bs in BS_GRID:
            name = f"ot33_grid_lr{lr:.0e}_bs{bs}".replace("+", "")
            configs.append({
                "name": name, "input_mode": "image_text", "selection_metric": "recall10",
                "lr": lr, "batch_size": bs, "patience": 5, "max_epochs": 40, "save_checkpoint": False,
            })
    return configs


def load_existing():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save(results):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results.values()), f, indent=2)


def main():
    configs = build_configs()
    results = load_existing()
    pending = [c for c in configs if c["name"] not in results]

    if pending:
        train_one = modal.Function.from_name("phase33-csanet-parity", "train_one")
        print(f"Launching {len(pending)} configs via .map()...")
        try:
            for cfg, result in zip(pending, train_one.map(pending)):
                print(f"{cfg['name']}: lr={cfg['lr']:.0e} bs={cfg['batch_size']} "
                      f"val_recall10={result['best_recall10']:.4f} best_epoch={result['best_epoch']} "
                      f"n_epochs_run={result['n_epochs_run']} wall_time={result['wall_time_sec']:.0f}s")
                results[cfg["name"]] = result
                save(results)
        except Exception as e:
            print(f"  .map() raised: {e!r} -- re-run this script; finished configs are saved.")

    print(f"\n{len(results)}/{len(configs)} configs done. Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

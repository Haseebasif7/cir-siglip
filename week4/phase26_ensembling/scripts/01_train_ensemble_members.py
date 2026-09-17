"""
Phase 26, step 1 (+ prep for step 5): train the additional independent
copies needed for ensembling. Reuses phase 25's deployed Modal app/function
("phase25-scale", train_one) and volume unchanged -- no new app, no
re-upload of base data.

Winning architecture from phase 25 (unchanged): hidden_dims=[1024],
out_dim=128, lr=0.0005, batch_size=256, weight_decay=0.0, tau=0.15, r_neg=8.
Model selection within each run stays keyed to validation-benchmark
Recall@10 (never validation loss), same as every phase in this sequence.

Ten independent copies are wanted for the main same-architecture ensemble.
Phase 25 already trained and checkpointed exactly this architecture at
seed=42 (the "final_scaled" confirmatory retrain -- val_recall10=0.16557,
best_epoch=1, checkpoint at week4/phase25_scale/models/final_scaled.pt) --
reusing that run as ensemble member 1 avoids retraining an identical
config from scratch and gives step 6's single best scale-phase model a
direct role inside the ensemble. Nine new seeds (1-9) are trained here to
reach ten total.

Step 5 (architectural diversity, secondary/smaller check) needs a couple of
width=512 checkpoints. Phase 25's own width sweep never saved checkpoints
(save_checkpoint defaulted to False there -- it was a validation-only
comparison), so two fresh width=512 runs are trained here too, at phase
25's own winning width=512 setting (lr=0.0005, val_recall10=0.1609), seeds
201/202 (distinct range, so they're never confused with the main pool).

All 11 new runs (9 width=1024 seeds + 2 width=512 diversity seeds) are
launched together in one Modal .map() call.
"""
import json
import shutil
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE25_DIR = REPO_ROOT / "week4/phase25_scale"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = DATA_DIR / "train_seeds_results.json"

MAIN_SEEDS = list(range(1, 10))          # 1..9 -- plus the reused seed=42 run = 10 total
DIVERSITY_SEEDS = [201, 202]             # width=512 diversity-check members
MAX_EPOCHS = 12
PATIENCE = 4


def build_configs():
    configs = []
    for seed in MAIN_SEEDS:
        configs.append({
            "name": f"ensemble_seed{seed}",
            "hidden_dims": [1024],
            "out_dim": 128,
            "lr": 0.0005,
            "seed": seed,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": 1,
            "save_checkpoint": True,
        })
    for seed in DIVERSITY_SEEDS:
        configs.append({
            "name": f"diversity_width512_seed{seed}",
            "hidden_dims": [512],
            "out_dim": 128,
            "lr": 0.0005,
            "seed": seed,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": 1,
            "save_checkpoint": True,
        })
    return configs


def download_checkpoint(remote_name, local_path):
    subprocess.run(
        ["modal", "volume", "get", "phase23-hp-tuning-data", remote_name, str(local_path), "--force"],
        check=True,
    )


def main():
    # Reuse phase 25's seed=42 run as ensemble member 1 -- copy checkpoint + result in.
    seed42_ckpt_src = PHASE25_DIR / "models" / "final_scaled.pt"
    seed42_ckpt_dst = MODELS_DIR / "ensemble_seed42.pt"
    shutil.copy(seed42_ckpt_src, seed42_ckpt_dst)
    with open(PHASE25_DIR / "data" / "final_config_train_result.json") as f:
        seed42_result = json.load(f)
    seed42_result = dict(seed42_result)
    seed42_result["reused_from"] = "week4/phase25_scale/data/final_config_train_result.json"
    seed42_result["local_checkpoint"] = str(seed42_ckpt_dst.relative_to(REPO_ROOT))

    train_one = modal.Function.from_name("phase25-scale", "train_one")
    configs = build_configs()
    print(f"Launching {len(configs)} new training runs in parallel on Modal "
          f"({len(MAIN_SEEDS)} width=1024 ensemble seeds + {len(DIVERSITY_SEEDS)} "
          f"width=512 diversity seeds)...")
    results = list(train_one.map(configs))

    all_results = [seed42_result] + results

    for r in all_results:
        ckpt_path = r.get("checkpoint_path")
        if ckpt_path and "local_checkpoint" not in r:
            remote_name = Path(ckpt_path).name
            local_path = MODELS_DIR / f"{r['config']['name']}.pt"
            download_checkpoint(remote_name, local_path)
            r["local_checkpoint"] = str(local_path.relative_to(REPO_ROOT))
            print(f"Downloaded {remote_name} -> {local_path}")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    for r in sorted(all_results, key=lambda r: -r["best_recall10"]):
        c = r["config"]
        print(f"{c['name']} (hidden_dims={c['hidden_dims']}, seed={c['seed']}): "
              f"val_recall10={r['best_recall10']:.4f} @epoch{r['best_epoch']}")


if __name__ == "__main__":
    main()

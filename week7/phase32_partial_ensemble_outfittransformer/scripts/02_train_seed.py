"""
Phase 32, step 2: trains ONE seed at a time (per the brief's explicit
instruction: no batched .map() calls, sequential only, with a real balance
check before each launch). Usage: `python3 02_train_seed.py <seed>`.

Exact same winning configuration as phase 31's seed 42
(ot31_budget_check_full.pt): input_mode=image_text, lr=1.5e-4,
batch_size=384, uniformity_weight=0.1, margin=0.2, selection_metric=recall10,
patience=25, max_epochs=100 -- differing ONLY in seed. Credit-safety
(checkpoint-persist-on-improvement, idempotent skip, warm-start resume) is
unchanged from phase 31's modal_app.py, copied verbatim.
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_balance import check_balance

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = DATA_DIR / "seed_results.json"

WINNING_CONFIG = {
    "input_mode": "image_text", "selection_metric": "recall10",
    "lr": 1.5e-4, "batch_size": 384, "uniformity_weight": 0.1, "margin": 0.2,
    "max_epochs": 100, "patience": 25, "save_checkpoint": True,
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 02_train_seed.py <seed>")
        sys.exit(1)
    seed = int(sys.argv[1])
    name = f"ot32_seed{seed}"

    results = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            results = {r["config"]["name"]: r for r in json.load(f)}

    if name in results and (MODELS_DIR / f"{name}.pt").exists():
        print(f"{name}: already completed (val_recall10={results[name]['best_recall10']:.4f}), skipping.")
        return

    balance = check_balance(f"before {name}")
    with open(DATA_DIR / f"balance_before_{name}.json", "w") as f:
        json.dump(balance, f, indent=2)
    if not balance["can_proceed"]:
        print(f"STOPPING: remaining balance ${balance['remaining']:.2f} is below the $1.50 safety margin. "
              f"Not launching {name}. Report the ensemble with whatever seeds already completed.")
        sys.exit(2)

    cfg = {"name": name, "seed": seed, **WINNING_CONFIG}
    train_one = modal.Function.from_name("phase32-partial-ensemble-ot", "train_one")
    print(f"Launching {name} (single .remote() call, not batched)...")
    result = train_one.remote(cfg)
    print(f"{name}: val_recall10={result['best_recall10']:.4f} best_epoch={result['best_epoch']} "
          f"n_epochs_run={result['n_epochs_run']} wall_time={result['wall_time_sec']:.0f}s")

    ckpt_path = result.get("checkpoint_path")
    if ckpt_path:
        remote_name = Path(ckpt_path).name
        local_ckpt = MODELS_DIR / f"{name}.pt"
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_name, str(local_ckpt), "--force"],
                        check=True)
        result["local_checkpoint"] = str(local_ckpt.relative_to(BASE_DIR.parent.parent))
        print(f"  downloaded -> {local_ckpt}")

    results[name] = result
    with open(OUT_JSON, "w") as f:
        json.dump(list(results.values()), f, indent=2)
    print(f"Saved {OUT_JSON}")

    balance_after = check_balance(f"after {name}")
    with open(DATA_DIR / f"balance_after_{name}.json", "w") as f:
        json.dump(balance_after, f, indent=2)


if __name__ == "__main__":
    main()

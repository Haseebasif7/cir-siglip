"""
Phase 34, step 3: seeds 1 and 2, sequential, local -- matching phase 33's
own 3-seed ensemble size exactly (42 + 1 + 2). Triggered only because step
2's gate cleared decisively (+49.7% relative vs. phase 33's mined-negative
seed 42). Runs entirely on the M4, no Modal budget involved -- same as
step 2.

Usage: `python3 02_train_additional_seeds.py <seed>` (run once per seed,
same discipline as phase 32/33's Modal scripts even though this is local:
one seed per invocation, results persisted incrementally so a re-run after
an interruption doesn't redo finished seeds).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_core

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "seed_results.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Identical to step 2's CONFIG (phase 33's winning hyperparameters) -- only
# the seed changes.
CONFIG = {"lr": 1e-4, "batch_size": 48, "patience": 5, "max_epochs": 40}


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 02_train_additional_seeds.py <seed>")
        sys.exit(1)
    seed = int(sys.argv[1])
    name = f"csanet34_seed{seed}"

    results = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            results = json.load(f)

    if name in results and (MODELS_DIR / f"{name}.pt").exists():
        print(f"{name}: already completed (val_recall10={results[name]['best_recall10']:.4f}), skipping.")
        return

    print(f"Loading catalog (device={DEVICE}, random same-category negatives)...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA,
                                        text_embeddings_npz=TEXT_EMBEDDINGS_NPZ, device=DEVICE)
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    val_benchmark = (bench["pools"], bench["queries"])

    print(f"Training seed={seed}, lr={CONFIG['lr']}, batch_size={CONFIG['batch_size']}, "
          f"max_epochs={CONFIG['max_epochs']}, patience={CONFIG['patience']} -- random same-category negatives...")
    t0 = time.time()
    result = train_core.run_training(
        catalog, DEVICE, max_epochs=CONFIG["max_epochs"], batch_size=CONFIG["batch_size"],
        lr=CONFIG["lr"], patience=CONFIG["patience"], seed=seed,
        selection_metric="recall10", val_benchmark=val_benchmark, log_every=1000,
    )
    wall_time = time.time() - t0
    print(f"\nseed={seed}: best_epoch={result['best_epoch']} val_recall10={result['best_recall10']:.4f} "
          f"n_epochs_run={result['n_epochs_run']} wall_time={wall_time:.0f}s n_params={result['n_params']}")

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / f"{name}.pt"
    if result["best_state"] is not None:
        torch.save(result["best_state"], ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    results[name] = {
        "config": {**CONFIG, "seed": seed}, "best_epoch": result["best_epoch"],
        "best_recall10": result["best_recall10"], "n_epochs_run": result["n_epochs_run"],
        "wall_time_sec": wall_time, "n_params": result["n_params"],
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

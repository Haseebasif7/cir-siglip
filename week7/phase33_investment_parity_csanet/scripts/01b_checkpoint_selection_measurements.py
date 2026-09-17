"""
Phase 33, step 1: A1 (val_loss selection, phase 13b's exact original config)
and A2 (recall10 selection) measurement runs, local (free). Usage:
`python3 01b_checkpoint_selection_measurements.py A1` or `A2` or `A2 <patience>`.

Phase 13b's original hyperparameters: lr=5e-5, batch_size=96, max_epochs=40,
patience=5 -- all held fixed for A1/A2 (only selection_metric differs, per
phase 31's own precedent of isolating one variable at a time).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_core

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEG_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"

OUT_JSON = BASE_DIR / "data" / "checkpoint_selection_results.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "A1"
    patience = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Loading catalog (device={DEVICE})...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA, NEG_CANDIDATES, device=DEVICE)
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    val_benchmark = (bench["pools"], bench["queries"])

    selection_metric = "val_loss" if run_id == "A1" else "recall10"
    print(f"Running {run_id}: selection_metric={selection_metric}, patience={patience}, "
          f"lr=5e-5, batch_size=96, max_epochs=40, seed=42")

    result = train_core.run_training(
        catalog, DEVICE, max_epochs=40, batch_size=96, lr=5e-5, patience=patience,
        seed=42, selection_metric=selection_metric, val_benchmark=val_benchmark, log_every=300,
    )

    print(f"\n{run_id}: best_epoch={result['best_epoch']} val_recall10={result['best_recall10']:.4f} "
          f"n_epochs_run={result['n_epochs_run']} wall_time={result['wall_time_sec']:.0f}s")
    for row in result["curve"]:
        print(" ", row)

    results = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            results = json.load(f)
    key = f"{run_id}_patience{patience}"
    results[key] = {
        "run_id": run_id, "selection_metric": selection_metric, "patience": patience,
        "best_epoch": result["best_epoch"], "best_recall10": result["best_recall10"],
        "n_epochs_run": result["n_epochs_run"], "wall_time_sec": result["wall_time_sec"],
        "curve": result["curve"],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")

    if result["best_state"] is not None:
        ckpt_path = BASE_DIR / "models" / f"{key}.pt"
        ckpt_path.parent.mkdir(exist_ok=True)
        torch.save(result["best_state"], ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()

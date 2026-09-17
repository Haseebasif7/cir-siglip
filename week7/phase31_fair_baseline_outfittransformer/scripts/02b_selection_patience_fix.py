"""
Phase 31, step 2 (correction, still within step 2's own scope): A2 and A3
(recall10-based selection, patience=8 carried over unchanged from phase
14b's val_loss-tuned value) both early-stopped at epoch ~9, val_recall10
~0.016-0.017 -- far below A1's (val_loss selection, ran the full 100 epochs)
eventual 0.0637 at epoch 93.

Diagnosis, direct from A1's own curve (data/selection_and_text_results.json
-- A1 tracks recall10 every epoch too, just doesn't select on it): recall10
has a real ~17-epoch plateau/dip (epochs 9-25, oscillating 0.0126-0.0169,
even dipping BELOW the epoch-9 value) before resuming a long, steady climb.
Patience=8 cannot survive this plateau when applied directly to recall10 --
this is not evidence the selection-metric fix is wrong, it's that the
patience budget (tuned against val_loss's much smoother curve) doesn't
transfer to recall10's noisier one. Fixing this is part of correctly
implementing "select by recall10" (step 2's own instruction), not
hyperparameter tuning (step 3) -- the patience value itself isn't being
optimized here, just set large enough (25, comfortably above the observed
17-epoch plateau, with margin) that early stopping reflects genuine
convergence rather than transient noise.

Re-runs A2 and A3 (recall10 selection) at patience=25, max_epochs=100
(unchanged budget cap, matching phase 14b's own). A1 (val_loss selection)
is NOT re-run -- it already used its own original patience=8 faithfully and
completed all 100 epochs without early stopping, so there's nothing to fix
there for the fidelity check itself.
"""
import json
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = DATA_DIR / "selection_and_text_results.json"

CONFIGS = [
    {"name": "ot31_repro_recall_p25", "input_mode": "image", "selection_metric": "recall10",
     "max_epochs": 100, "patience": 25, "save_checkpoint": True},
    {"name": "ot31_text_recall_p25", "input_mode": "image_text", "selection_metric": "recall10",
     "max_epochs": 100, "patience": 25, "save_checkpoint": True},
]


def load_existing_results():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save_results(results_by_name):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results_by_name.values()), f, indent=2)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    results_by_name = load_existing_results()

    pending = [cfg for cfg in CONFIGS
               if cfg["name"] not in results_by_name or not (MODELS_DIR / f"{cfg['name']}.pt").exists()]

    if pending:
        train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")
        print(f"Launching {len(pending)} runs (patience=25) in parallel via .map()...")
        try:
            for cfg, result in zip(pending, train_one.map(pending)):
                name = cfg["name"]
                print(f"{name}: best_epoch={result['best_epoch']} "
                      f"val_recall10={result['best_recall10']:.4f} "
                      f"wall_time={result['wall_time_sec']:.0f}s n_epochs_run={result['n_epochs_run']}")
                ckpt_path = result.get("checkpoint_path")
                if ckpt_path:
                    remote_name = Path(ckpt_path).name
                    local_ckpt = MODELS_DIR / f"{name}.pt"
                    subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_name,
                                     str(local_ckpt), "--force"], check=True)
                    result["local_checkpoint"] = str(local_ckpt.relative_to(BASE_DIR.parent.parent))
                    print(f"  downloaded -> {local_ckpt}")
                results_by_name[name] = result
                save_results(results_by_name)
        except Exception as e:
            print(f"  .map() raised: {e!r} -- re-run this script.")

    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()

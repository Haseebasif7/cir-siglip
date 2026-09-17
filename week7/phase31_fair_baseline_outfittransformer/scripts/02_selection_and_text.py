"""
Phase 31, step 2: the three isolated measurement runs, all at phase 14b's
exact original hyperparameters (lr=2e-5, batch_size=96, max_epochs=100,
patience=8, margin=0.3, num_negatives=10, uniformity_weight=1.0), seed 42:

  A1 (ot31_repro_valloss):  input=image,      selection=val_loss  -- Modal port fidelity vs. phase 14b
  A2 (ot31_repro_recall):   input=image,      selection=recall10  -- the selection fix, alone
  A3 (ot31_text_recall):    input=image_text, selection=recall10  -- + text, alone

Launched together via .map() for wall-clock parallelism; each result
persisted to disk and its checkpoint downloaded the moment it finishes
(credit-safety pattern, phase 27-30 precedent).
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
    {"name": "ot31_repro_valloss", "input_mode": "image", "selection_metric": "val_loss",
     "max_epochs": 100, "patience": 8, "save_checkpoint": True},
    {"name": "ot31_repro_recall", "input_mode": "image", "selection_metric": "recall10",
     "max_epochs": 100, "patience": 8, "save_checkpoint": True},
    {"name": "ot31_text_recall", "input_mode": "image_text", "selection_metric": "recall10",
     "max_epochs": 100, "patience": 8, "save_checkpoint": True},
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
    for cfg in CONFIGS:
        if cfg not in pending:
            print(f"{cfg['name']}: already completed locally "
                  f"(val_recall10={results_by_name[cfg['name']]['best_recall10']:.4f}), skipping.")

    if pending:
        train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")
        print(f"Launching {len(pending)} runs in parallel via .map()...")
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
            print(f"  .map() raised: {e!r} -- credits or an interruption likely. "
                  f"Re-run this script; already-finished runs are already saved above, "
                  f"and train_one's own idempotent/warm-start logic covers the rest.")

    print(f"\n{len(results_by_name)}/{len(CONFIGS)} runs available. Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

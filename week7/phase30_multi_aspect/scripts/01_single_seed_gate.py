"""
Phase 30, step 3: single-seed gate. Trains seed=42 multi-aspect to
convergence and compares against phase 27/28's own single-seed text-only
result (val Recall@10 = 0.18194, seed=42, mean-pool) before spending any
further budget. Per the brief's exact go/no-go rule: proceed to the full
ensemble only if this run beats that baseline by at least 2% relative on
validation Recall@10; otherwise stop, report, and treat this as the answer.
"""
import json
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = DATA_DIR / "single_seed_result.json"

# Phase 27's own text_only run, seed=42, val Recall@10 -- see
# week7/phase27_text_and_category/data/train_variants_results.json and
# results_table.md. Same baseline phase 29 used for its own gate.
PHASE28_BASELINE_R10 = 0.18194290772294755
GATE_RELATIVE_GAIN = 0.02  # brief's 2% threshold


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    cfg = {
        "name": "multiaspect_seed42",
        "seed": 42,
        "max_epochs": 12,
        "patience": 4,
        "save_checkpoint": True,
    }

    train_one = modal.Function.from_name("phase30-multi-aspect", "train_one")
    print(f"Launching single-seed gate run ({cfg['name']})...")
    result = train_one.remote(cfg)

    r10 = result["best_recall10"]
    relative_gain = r10 / PHASE28_BASELINE_R10 - 1
    go = relative_gain >= GATE_RELATIVE_GAIN

    print(f"best_epoch={result['best_epoch']} val_recall10={r10:.4f} "
          f"wall_time={result['wall_time_sec']:.0f}s")
    print(f"Baseline (phase 27/28 text_only, seed=42): {PHASE28_BASELINE_R10:.4f}")
    print(f"Relative gain: {relative_gain:+.1%}  ->  {'GO' if go else 'NO-GO'}")

    ckpt_path = result.get("checkpoint_path")
    if ckpt_path:
        remote_name = Path(ckpt_path).name
        local_ckpt = MODELS_DIR / "multiaspect_seed42.pt"
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_name, str(local_ckpt), "--force"],
                        check=True)
        print(f"  downloaded -> {local_ckpt}")

    out = {
        "result": result,
        "baseline_r10": PHASE28_BASELINE_R10,
        "relative_gain": relative_gain,
        "gate_threshold": GATE_RELATIVE_GAIN,
        "decision": "GO" if go else "NO-GO",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

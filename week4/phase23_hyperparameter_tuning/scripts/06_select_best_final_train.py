"""
Phase 23, pre-step-7: gather every configuration tried across steps 3-6
(all still scored on the VALIDATION benchmark only), pick the single overall
best by validation Recall@10, then train that exact configuration one more
time (generous epoch/patience budget, checkpoint saved) so there's a clean
final checkpoint to hand to step 7's one-time test-benchmark evaluation --
rather than reusing whichever intermediate run happened to already produce
that config's number, which could differ run to run due to only the
random-negative-sampling stream (weights/data order are seeded, but which
specific negatives get sampled depends on the RNG stream position, which
differs slightly run to run because of the surrounding grid).
"""
import json
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_FILES = {
    "step3_lr_bs": BASE_DIR / "data" / "lr_bs_grid_results.json",
    "step4_tau": BASE_DIR / "data" / "temperature_sweep_results.json",
    "step5_long": BASE_DIR / "data" / "long_patience_result.json",
    "step6_refine": BASE_DIR / "data" / "refinement_results.json",
}
OUT_JSON = BASE_DIR / "data" / "final_config_train_result.json"
CHECKPOINT_LOCAL = BASE_DIR / "models" / "final_tuned.pt"


def load_all_runs():
    runs = []
    for stage, path in RESULT_FILES.items():
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for r in data:
                runs.append((stage, r))
        else:
            runs.append((stage, data))
    return runs


def main():
    runs = load_all_runs()
    stage, best = max(runs, key=lambda sr: sr[1]["best_recall10"])
    cfg = dict(best["config"])
    print(f"Overall best across all stages: stage={stage} name={cfg['name']} "
          f"val_recall10={best['best_recall10']:.4f}")

    train_one = modal.Function.from_name("phase23-hp-tuning", "train_one")

    # Generous budget for this final confirmatory run -- at least as long as
    # step 5's own budget, so the final checkpoint isn't accidentally cut off
    # earlier than the long-patience check already showed was needed.
    final_config = {
        **cfg,
        "name": "final_tuned",
        "max_epochs": max(cfg.get("max_epochs", 12), 60),
        "patience": max(cfg.get("patience", 4), 15),
        "eval_every": 1,
        "save_checkpoint": True,
    }
    print(f"Final training run config: {final_config}")

    result = train_one.remote(final_config)
    result["selected_from_stage"] = stage
    result["selected_from_name"] = cfg["name"]

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT_JSON}")
    print(f"Final tuned checkpoint best epoch: {result['best_epoch']}, "
          f"val_recall10={result['best_recall10']:.4f}")

    ckpt_path = result.get("checkpoint_path")
    if ckpt_path:
        remote_name = Path(ckpt_path).name
        CHECKPOINT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["modal", "volume", "get", "phase23-hp-tuning-data", remote_name, str(CHECKPOINT_LOCAL), "--force"],
            check=True,
        )
        print(f"Downloaded checkpoint to {CHECKPOINT_LOCAL}")
    else:
        print("WARNING: no checkpoint_path in result -- checkpoint was not saved.")


if __name__ == "__main__":
    main()

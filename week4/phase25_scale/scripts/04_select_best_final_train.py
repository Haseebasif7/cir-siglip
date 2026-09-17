"""
Phase 25, pre-step-6: gather every configuration tried across steps 1-3
(width, depth, embedding dimension -- all still scored on the VALIDATION
benchmark only), pick the single overall best by validation Recall@10
across the whole scale search INCLUDING the already-established phase 23/24
baseline (width=256, 1 hidden layer, out_dim=128, lr=0.001: val Recall@10 =
0.1600, bit-exactly reproduced three times already) -- scale is only worth
adopting if it beats that baseline, not merely if some larger architecture
scores well in isolation. If the baseline wins, no new training run is
needed; report that directly. If a scaled config wins, retrain it once more
(generous budget, checkpoint saved) for step 6's one-time test-benchmark
check.
"""
import json
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_FILES = {
    "step1_width": BASE_DIR / "data" / "width_sweep_results.json",
    "step2_depth": BASE_DIR / "data" / "depth_sweep_results.json",
    "step3_embedding_dim": BASE_DIR / "data" / "embedding_dim_sweep_results.json",
}
OUT_JSON = BASE_DIR / "data" / "final_config_train_result.json"
CHECKPOINT_LOCAL = BASE_DIR / "models" / "final_scaled.pt"

BASELINE_HIDDEN_DIMS = [256]
BASELINE_OUT_DIM = 128
BASELINE_LR = 0.001
BASELINE_RECALL10 = 0.1600  # phase 23/24's established, triple-reproduced number


def load_all_runs():
    runs = []
    for stage, path in RESULT_FILES.items():
        with open(path) as f:
            data = json.load(f)
        for r in data:
            runs.append((stage, r))
    return runs


def main():
    runs = load_all_runs()
    stage, best = max(runs, key=lambda sr: sr[1]["best_recall10"])
    cfg = dict(best["config"])

    print(f"Best scaled config across steps 1-3: stage={stage} name={cfg['name']} "
          f"hidden_dims={cfg['hidden_dims']} out_dim={cfg['out_dim']} lr={cfg['lr']} "
          f"val_recall10={best['best_recall10']:.4f}")
    print(f"Phase 23/24 baseline (hidden_dims={BASELINE_HIDDEN_DIMS}, out_dim={BASELINE_OUT_DIM}, "
          f"lr={BASELINE_LR}): val_recall10={BASELINE_RECALL10:.4f}")

    if best["best_recall10"] <= BASELINE_RECALL10:
        print("Baseline (width=256, depth=1, out_dim=128) is still the best configuration -- "
              "scale did not improve on it. No new confirmatory training run or test-benchmark "
              "check is warranted; the current best_tuned.pt checkpoint stays the reference.")
        result = {
            "outcome": "baseline_wins",
            "baseline_config": {"hidden_dims": BASELINE_HIDDEN_DIMS, "out_dim": BASELINE_OUT_DIM,
                                 "lr": BASELINE_LR, "val_recall10": BASELINE_RECALL10},
            "best_scaled_config": cfg,
            "best_scaled_val_recall10": best["best_recall10"],
            "selected_from_stage": stage,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {OUT_JSON}")
        return

    train_one = modal.Function.from_name("phase25-scale", "train_one")

    final_config = {
        **cfg,
        "name": "final_scaled",
        "max_epochs": max(cfg.get("max_epochs", 12), 60),
        "patience": max(cfg.get("patience", 4), 15),
        "eval_every": 1,
        "save_checkpoint": True,
    }
    print(f"Scale beats the baseline -- confirmatory retrain config: {final_config}")

    result = train_one.remote(final_config)
    result["outcome"] = "scale_wins"
    result["selected_from_stage"] = stage
    result["selected_from_name"] = cfg["name"]
    result["baseline_val_recall10"] = BASELINE_RECALL10

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT_JSON}")
    print(f"Final scaled checkpoint best epoch: {result['best_epoch']}, "
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

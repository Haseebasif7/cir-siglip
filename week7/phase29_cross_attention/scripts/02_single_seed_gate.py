"""
Phase 29, step 3: train a single seed (42, matching phase 27/28's own
convention) to convergence at the LR chosen in step 1, evaluate on the
validation benchmark, and apply the brief's explicit stop/go gate before
spending a ten-seed budget: continue to full ensemble only if val Recall@10
beats phase 28's single-seed text-only result (0.1819, seed=42) by at least
2 percent relative.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUT_JSON = DATA_DIR / "single_seed_result.json"
OUT_MD = BASE_DIR / "single_seed_gate.md"

PHASE28_BASELINE_R10 = 0.1819  # phase 27/28's seed=42 text-only, val benchmark
GATE_RELATIVE_GAIN = 0.02

with open(DATA_DIR / "lr_check_results.json") as f:
    lr_results = json.load(f)
CHOSEN_LR = 0.0005  # winner of step 1's check -- see training_log.md


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    train_one = modal.Function.from_name("phase29-cross-attention", "train_one")

    cfg = {"name": "xattn_seed42", "lr": CHOSEN_LR, "seed": 42, "max_epochs": 12,
           "patience": 4, "eval_every": 1, "save_checkpoint": True}

    print(f"Training seed=42 at lr={CHOSEN_LR}...")
    result = train_one.remote(cfg)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    r10 = result["best_recall10"]
    print(f"seed=42: best_epoch={result['best_epoch']} val_recall10={r10:.4f} "
          f"wall_time={result['wall_time_sec']:.0f}s")

    ckpt_path = result.get("checkpoint_path")
    if ckpt_path:
        import subprocess
        local_ckpt = MODELS_DIR / "xattn_seed42.pt"
        local_xattn_ckpt = MODELS_DIR / "xattn_seed42_xattn.pt"
        subprocess.run(["modal", "volume", "get", "phase27-text-category-data",
                         Path(ckpt_path).name, str(local_ckpt), "--force"], check=True)
        subprocess.run(["modal", "volume", "get", "phase27-text-category-data",
                         Path(result["xattn_checkpoint_path"]).name, str(local_xattn_ckpt),
                         "--force"], check=True)
        print(f"downloaded -> {local_ckpt}, {local_xattn_ckpt}")

    relative_gain = (r10 - PHASE28_BASELINE_R10) / PHASE28_BASELINE_R10
    threshold_r10 = PHASE28_BASELINE_R10 * (1 + GATE_RELATIVE_GAIN)
    go = r10 >= threshold_r10

    lines = ["# Phase 29: Single-Seed Gate (Step 3)\n",
              "| Configuration | Val Recall@10 |",
              "|---|---|",
              f"| Phase 28 text-only, single seed (42), mean-pool | {PHASE28_BASELINE_R10:.4f} |",
              f"| Phase 29 cross-attention, single seed (42) | {r10:.4f} |",
              "",
              f"Relative change: {relative_gain*100:+.2f}% "
              f"(gate requires >= {GATE_RELATIVE_GAIN*100:.0f}% to proceed to full ensemble; "
              f"threshold = {threshold_r10:.4f}).",
              "",
              f"**Decision: {'GO -- proceed to full 10-seed ensemble (step 4).' if go else 'NO-GO -- stop before the ten-seed budget, per the brief explicit discipline. Cross-attention does not show a real single-seed signal over mean-pooling; the phase stops here and reports this as the answer.'}**",
              "",
              f"Learning rate used: {CHOSEN_LR} (chosen in step 1's LR check over "
              f"phase 28's original 0.001 -- see training_log.md).",
              f"Epochs to convergence: {result['best_epoch'] + 1} (best at epoch {result['best_epoch']}, "
              f"early stopping patience=4).",
              f"Wall time: {result['wall_time_sec']:.0f}s.",
              ""]

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"\nGate decision: {'GO' if go else 'NO-GO'}. Wrote {OUT_JSON} and {OUT_MD}")
    return go


if __name__ == "__main__":
    main()

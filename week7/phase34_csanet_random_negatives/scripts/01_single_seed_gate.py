"""
Phase 34, step 2: the single-seed gate. Trains ONE seed (42) at phase 33's
exact winning configuration (image+text, recall10 selection, lr=1e-4,
batch_size=48, patience=5, max_epochs=40), with random same-category
negatives instead of mined ones -- the only variable under test. Compares
directly against phase 33's own seed-42 validation result at the identical
configuration (0.1075, `phase33/scripts/03_tuning_grid.py` /
`04_train_seed.py` seed_results.json).

Gate: >= 5% relative improvement at validation R@10 to proceed to
ensembling (step 3). A higher bar than phases 29/30's 2% (this is a
training-PROCEDURE change, not an architecture change, and this project's
own procedure changes have historically shown large or null effects, not
small incremental ones -- see phase34.md's own framing). If the gate isn't
cleared, stop here and report a null/negative finding -- no ensembling, no
test-benchmark touch.
"""
import json
import time
from pathlib import Path

import torch

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
OUT_JSON = DATA_DIR / "seed42_gate_result.json"
OUT_MD = BASE_DIR / "single_seed_gate.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Phase 33's exact winning configuration (03_tuning_grid.py / 04_train_seed.py),
# reused unchanged -- see negative_sampling_change.md for why this phase does
# not re-tune under the new negative-sampling scheme.
CONFIG = {"lr": 1e-4, "batch_size": 48, "patience": 5, "max_epochs": 40, "seed": 42}
PHASE33_SEED42_VAL_R10 = 0.1075  # phase33/scripts/03_tuning_grid.py winner, reproduced exactly by 04_train_seed.py
GATE_THRESHOLD = 0.05  # 5% relative, per this phase's own brief


def main():
    print(f"Loading catalog (device={DEVICE}, random same-category negatives)...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA,
                                        text_embeddings_npz=TEXT_EMBEDDINGS_NPZ, device=DEVICE)
    print(f"  in_dim={catalog['in_dim']}")
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    val_benchmark = (bench["pools"], bench["queries"])

    print(f"Training seed=42, lr={CONFIG['lr']}, batch_size={CONFIG['batch_size']}, "
          f"max_epochs={CONFIG['max_epochs']}, patience={CONFIG['patience']} "
          f"-- random same-category negatives...")
    t0 = time.time()
    result = train_core.run_training(
        catalog, DEVICE, max_epochs=CONFIG["max_epochs"], batch_size=CONFIG["batch_size"],
        lr=CONFIG["lr"], patience=CONFIG["patience"], seed=CONFIG["seed"],
        selection_metric="recall10", val_benchmark=val_benchmark, log_every=1000,
    )
    wall_time = time.time() - t0
    print(f"\nseed=42: best_epoch={result['best_epoch']} val_recall10={result['best_recall10']:.4f} "
          f"n_epochs_run={result['n_epochs_run']} wall_time={wall_time:.0f}s n_params={result['n_params']}")
    for row in result["curve"]:
        print(" ", row)

    val_r10 = result["best_recall10"]
    rel_gain = val_r10 / PHASE33_SEED42_VAL_R10 - 1
    gate_cleared = rel_gain >= GATE_THRESHOLD

    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    out = {
        "config": CONFIG, "val_recall10": val_r10, "phase33_seed42_val_recall10": PHASE33_SEED42_VAL_R10,
        "relative_gain": rel_gain, "gate_threshold": GATE_THRESHOLD, "gate_cleared": gate_cleared,
        "best_epoch": result["best_epoch"], "n_epochs_run": result["n_epochs_run"],
        "wall_time_sec": wall_time, "n_params": result["n_params"], "curve": result["curve"],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_JSON}")

    if result["best_state"] is not None:
        ckpt_path = MODELS_DIR / "csanet34_seed42.pt"
        torch.save(result["best_state"], ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    lines = [
        "# Phase 34, Step 2: Single-Seed Gate\n",
        "Seed 42, phase 33's exact winning configuration (image+text, recall10 selection, lr=1e-4, "
        "batch_size=48, patience=5, max_epochs=40), random same-category negatives instead of mined "
        "candidates -- the only variable under test.\n",
        "## Result\n",
        f"- Phase 33 (mined negatives), seed 42, validation R@10: **{PHASE33_SEED42_VAL_R10:.4f}**",
        f"- Phase 34 (random negatives), seed 42, validation R@10: **{val_r10:.4f}**",
        f"- Relative gain: **{rel_gain:+.1%}**",
        f"- Gate (>= {GATE_THRESHOLD:.0%} relative to proceed to ensembling): "
        f"**{'CLEARED' if gate_cleared else 'NOT CLEARED'}**",
        f"- best_epoch={result['best_epoch']}, n_epochs_run={result['n_epochs_run']}, "
        f"wall_time={wall_time:.0f}s, n_params={result['n_params']}",
        "",
        "## Decision\n",
    ]
    if gate_cleared:
        lines.append(
            "Gate cleared. Proceeding to step 3: train 2 additional seeds (1, 2) and build the 3-seed "
            "ensemble, matching phase 33's own ensemble size exactly."
        )
    else:
        lines.append(
            "Gate NOT cleared. Per this phase's own brief, stopping here: no ensembling, no test-benchmark "
            "touch. CSA-Net's remaining gap (identified in phase 33) is reported as genuinely architectural "
            "rather than investment-driven -- see phase34_notes.md for the full interpretation."
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    print(f"\nGATE: {'CLEARED' if gate_cleared else 'NOT CLEARED'} ({rel_gain:+.1%} vs. {GATE_THRESHOLD:.0%} threshold)")


if __name__ == "__main__":
    main()

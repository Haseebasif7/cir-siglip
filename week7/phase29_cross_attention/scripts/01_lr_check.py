"""
Phase 29, step 2: small learning-rate check (current 0.001, half 0.0005,
double 0.002) before committing to the full single-seed run, per the
brief's own caution that carrying over hyperparameters unchanged has bitten
this project before. Short runs (6 epochs, patience 3), no checkpoint
saving -- this script only picks a learning rate, it doesn't produce a
usable model.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "lr_check_results.json"
OUT_MD = BASE_DIR / "training_log.md"

LRS = {"half": 0.0005, "current": 0.001, "double": 0.002}


def main():
    train_one = modal.Function.from_name("phase29-cross-attention", "train_one")

    configs = [
        {"name": f"lr_check_{tag}", "lr": lr, "max_epochs": 6, "patience": 3,
         "eval_every": 1, "save_checkpoint": False}
        for tag, lr in LRS.items()
    ]

    print(f"Launching {len(configs)} LR-check runs in parallel via .map()...")
    results = {}
    for cfg, result in zip(configs, train_one.map(configs)):
        tag = cfg["name"].replace("lr_check_", "")
        results[tag] = result
        print(f"{tag} (lr={cfg['lr']}): best_epoch={result['best_epoch']} "
              f"val_recall10={result['best_recall10']:.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    best_tag = max(results, key=lambda t: results[t]["best_recall10"])
    best_lr = LRS[best_tag]

    lines = ["# Phase 29: Training Log\n", "## Step 2: learning rate check\n",
             "Short runs (6 epochs, patience 3) at three learning rates, "
             "compared on the validation benchmark before committing to the "
             "full single-seed run.\n",
             "| LR | Tag | Best epoch | Val Recall@10 |",
             "|---|---|---|---|"]
    for tag, lr in LRS.items():
        r = results[tag]
        lines.append(f"| {lr} | {tag} | {r['best_epoch']} | {r['best_recall10']:.4f} |")
    lines.append("")
    if best_tag == "current":
        lines.append(f"**Chosen LR: {best_lr} (phase 28's original, carried over as-is)** "
                      f"-- the plain carryover won the check, so no override is needed. "
                      f"Per the brief: \"If plain carryover works fine, that's your answer, "
                      f"don't over-engineer this.\"")
    else:
        lines.append(f"**Chosen LR: {best_lr} ({best_tag})** -- beat phase 28's original "
                      f"lr=0.001 on this short check, so the full single-seed run in step 3 "
                      f"uses this value instead of a blind carryover.")
    lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"\nChosen LR: {best_lr} ({best_tag}). Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()

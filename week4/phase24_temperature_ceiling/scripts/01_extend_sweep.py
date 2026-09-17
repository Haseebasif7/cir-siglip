"""
Phase 24, step 1: extend phase 23's temperature sweep upward. Phase 23 found
Recall@10 rising monotonically from tau=0.03 to tau=0.15 (the edge of its
tested range) at lr=0.001/bs=256/wd=0.0/R=8, with no sign of a peak --
flagged there as an open gap, closed here.

Reuses the exact same Modal infrastructure phase 23 built and deployed
(app "phase23-hp-tuning", function "train_one") and the exact same
validation benchmark (phase 23's data/cir_val_benchmark.json, referenced
directly on the shared Modal volume -- not re-uploaded) -- no new
infrastructure needed, this phase's brief is purely about extending the
tau range with the same protocol (validation-benchmark Recall@10 selection,
never validation loss, test benchmark untouched until step 3).

Run as multiple small batches rather than one huge grid, since the stopping
rule ("stop once Recall@10 declines for a couple of consecutive steps or
clearly flattens") is inherently adaptive -- can't be decided before seeing
results. Call with --batch to pass a comma-separated list of tau values.
"""
import json
import sys
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_JSON = BASE_DIR / "data" / "tau_extension_results.json"

LR = 0.001
BS = 256
WD = 0.0
R_NEG = 8
MAX_EPOCHS = 12
PATIENCE = 4


def main():
    taus = [float(x) for x in sys.argv[1].split(",")]
    train_one = modal.Function.from_name("phase23-hp-tuning", "train_one")

    configs = []
    for tau in taus:
        configs.append({
            "name": f"phase24_tau{tau}",
            "lr": LR,
            "batch_size": BS,
            "weight_decay": WD,
            "tau": tau,
            "r_neg": R_NEG,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": 1,
        })

    print(f"Launching {len(configs)} tau configs in parallel on Modal: {taus}")
    results = list(train_one.map(configs))

    existing = []
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(OUT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {OUT_JSON} ({len(existing)} total configs so far)")

    for r in sorted(results, key=lambda r: r["config"]["tau"]):
        c = r["config"]
        print(f"tau={c['tau']}: best_recall10={r['best_recall10']:.4f} @epoch{r['best_epoch']} "
              f"({r['n_epochs_run']} epochs run, {r['wall_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()

"""
Phase 19, step 1: confirm raw SigLIP's existing Recall@10/30/50 numbers on
this project's CIR benchmark are still consistent, by actually re-running
the eval (cheap -- no training, just embedding similarity) rather than
blindly trusting the old citation (0.0553/0.1067/0.1437, first measured in
phase 12, re-cited in every comparison since including phase 14b's
results_table.md). Uses phase 9's own siglip_base.npz directly, unweighted
-- this is what "raw SigLIP" has meant in every prior comparison in this
project.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
OUT_MD = BASE_DIR / "data" / "siglip_baseline_confirmation.md"

CITED_RECALL = {10: 0.0553, 30: 0.1067, 50: 0.1437}


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms  # already ~unit norm as stored, renormalize defensively

    pools, queries = load_benchmark()
    print(f"Loaded benchmark: {len(queries)} queries, {len(pools)} category pools")

    recall, n_total, n_skipped = evaluate_recall(pools, queries, item_ids, embeddings)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    match = all(abs(recall[k] - CITED_RECALL[k]) < 1e-4 for k in CITED_RECALL)

    lines = [
        "# Phase 19, Step 1: Raw SigLIP Baseline Confirmation",
        "",
        "Re-ran the eval directly (no retraining -- this is a frozen, untrained "
        "baseline, it cannot have changed) on the identical CIR benchmark "
        "(`week4/phase12_controllable_modes/data/cir_benchmark.json`) used for "
        "every comparison in this redirected sequence.",
        "",
        "| | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Cited (phase 12 onward) | {CITED_RECALL[10]:.4f} | {CITED_RECALL[30]:.4f} | {CITED_RECALL[50]:.4f} |",
        f"| Re-measured here | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} |",
        "",
        f"n_total={n_total}, n_skipped={n_skipped}.",
        "",
        f"**{'CONFIRMED: exact match.' if match else 'MISMATCH -- investigate before proceeding.'}**",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    assert match, "Raw SigLIP baseline does not match the cited numbers -- stop and investigate."


if __name__ == "__main__":
    main()

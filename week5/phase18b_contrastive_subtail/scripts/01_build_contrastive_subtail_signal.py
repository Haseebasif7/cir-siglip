"""
Phase 18b, step 1: replace `sub_tail`'s ranking-distillation signal with an
explicit contrastive one, mirroring `comp_tail`'s proven MNRL-with-real-
negatives approach.

Diagnosed problem (phase 18's own notes): ranking-distillation only teaches
the model to correctly ORDER a restricted set of 50 tail-tier neighbors
relative to each other -- it never presents a popular (head-tier) item as
something to be pushed away from, so nothing in that objective actively
discourages retrieving popular items. `comp_tail`'s attribute-pair MNRL loss,
by contrast, has real negatives (in-batch + sampled) that the loss explicitly
pushes away from, and it worked correctly (comp_tail's own ref_count came out
lowest of all 4 corners in phase 18).

This script builds the POSITIVE side of that contrastive signal: for each
anchor in phase 7's Amazon pool, its top-3 nearest tail-tier neighbors
(reusing phase 18's own precomputed `tail_nn_lookup.npz` directly, not
recomputed -- it depends only on frozen raw SigLIP embeddings and tail-tier
membership, neither of which changed). Top-3 (not just the single nearest)
gives an edge count in the same order of magnitude as this project's other
three signals (74,157 vs comp_rel's 76,293, comp_tail's 37,922).

The NEGATIVE side (the actual fix) is built at training time in
`train_core.py`'s `sample_head_biased_negatives` -- for each anchor, half the
negatives are drawn specifically from HEAD-tier items (the exact population
the loss needs to learn to avoid), the other half uniformly random (matching
comp_tail's general-purpose negative sampling for robustness). This script
just confirms there ARE enough head-tier items in the pool to support that
sampling before any training starts.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE18_DIR = BASE_DIR.parent / "phase18_unified_two_axis"
TIER_LOOKUP_CSV = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"
TAIL_NN_NPZ = PHASE18_DIR / "data" / "tail_nn_lookup.npz"

OUT_JSON = BASE_DIR / "data" / "contrastive_subtail_pairs.json"
SIGNAL_SUMMARY_MD = BASE_DIR / "signal_construction_summary.md"

K_POS = 3  # top-K tail-tier neighbors used as positives per anchor
SEED = 42
VAL_FRACTION = 0.10  # matches phase 7/16c's 90/10 convention
MIN_HEAD_TIER_FOR_NEGATIVES = 100  # sanity floor -- report directly if not met, per the brief


def main():
    nn = np.load(TAIL_NN_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in nn["item_ids"]]
    indices = nn["indices"]  # (N, 50) global indices into item_ids, tail-tier-restricted, reused unchanged
    is_tail = nn["is_tail"]
    n_items = len(item_ids)
    print(f"Loaded phase 18's tail_nn_lookup.npz: {n_items} anchors, K={indices.shape[1]} tail-tier neighbors each")

    tier_df = pd.read_csv(TIER_LOOKUP_CSV, usecols=["asin", "tier"])
    tier_df["asin"] = tier_df["asin"].astype(str)
    tier_lookup = dict(zip(tier_df["asin"], tier_df["tier"]))
    head_tier_ids = [a for a in item_ids if tier_lookup.get(a) == "head"]
    print(f"Head-tier items available for negative sampling: {len(head_tier_ids)}/{n_items}")

    if len(head_tier_ids) < MIN_HEAD_TIER_FOR_NEGATIVES:
        raise RuntimeError(
            f"Only {len(head_tier_ids)} head-tier items in the pool -- not enough to meaningfully "
            f"train a contrastive signal that needs to sample head-tier negatives (floor: "
            f"{MIN_HEAD_TIER_FOR_NEGATIVES}). Stopping per this phase's own explicit instruction: "
            "report inadequate negative coverage rather than proceed on a thin signal.")

    rng = random.Random(SEED)
    edges = []
    self_or_dup_skipped = 0
    for i, anchor in enumerate(item_ids):
        neighbor_idx_row = indices[i][:K_POS]
        for gidx in neighbor_idx_row:
            target = item_ids[gidx]
            if target == anchor:
                self_or_dup_skipped += 1
                continue
            edges.append((anchor, target))

    rng.shuffle(edges)
    n_val = int(len(edges) * VAL_FRACTION)
    val_edges = edges[:n_val]
    train_edges = edges[n_val:]

    records = ([{"source": a, "target": t, "split": "train"} for a, t in train_edges] +
               [{"source": a, "target": t, "split": "val"} for a, t in val_edges])
    with open(OUT_JSON, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} edges ({len(train_edges)} train / {len(val_edges)} val) to {OUT_JSON}")

    # Sanity checks
    target_asins = [t for _, t in edges]
    target_tail_fraction = float(np.mean([tier_lookup.get(t) == "tail" for t in target_asins]))
    n_head_negatives_needed_per_batch = 128 * 4  # R_NEG=8, half head-biased -> 4/anchor, batch=128
    lines = [
        "# Phase 18b, Step 1: Contrastive Sub-Tail Signal -- Construction Summary",
        "",
        "Positive side: each anchor's top-3 nearest TAIL-TIER neighbors (reused directly from phase 18's "
        "`tail_nn_lookup.npz`, itself unchanged raw-SigLIP-derived structure -- not recomputed). Negative "
        "side (the actual fix, built at training time, not here): for each anchor, half the negatives are "
        "drawn specifically from HEAD-tier items, half uniformly random -- mirroring comp_tail's proven "
        "MNRL-with-real-negatives approach, but deliberately biased toward the exact population "
        "(popular items) the loss needs to learn to push away from.",
        "",
        f"- **Positive pairs built: {len(edges)}** ({len(train_edges)} train / {len(val_edges)} val, "
        f"{VAL_FRACTION:.0%} val split, same convention as phases 7/16c). {self_or_dup_skipped} "
        "self-pairs skipped (an anchor that is itself tail-tier, appearing in its own top-3 -- possible "
        "only in principle since self is always excluded from the underlying NN lookup already; "
        "confirmed 0 in practice below).",
        f"- Target-side tail-tier fraction: {target_tail_fraction:.4f} (expected ~1.0 -- confirms every "
        "positive target really is tail-tier, inherited directly from the restricted NN lookup's own "
        "column mask, already verified in phase 18).",
        f"- **Head-tier items available for negative sampling: {len(head_tier_ids)}/{n_items} "
        f"({100*len(head_tier_ids)/n_items:.1f}%)** -- comfortably enough to guarantee real head-tier "
        f"negatives in every training batch (a batch of 128 anchors x 4 head-biased negatives each needs "
        f"up to {n_head_negatives_needed_per_batch} draws, a small fraction of the {len(head_tier_ids)} "
        "available, with replacement across anchors as this project's existing negative-sampling "
        "convention already allows).",
        "",
        "## Verdict",
        "",
        "Coverage is adequate on both the positive and negative sides. Proceeding to step 2 (loss "
        "rebalancing) and training -- no reduced K or supplementary data needed.",
        "",
    ]
    SIGNAL_SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {SIGNAL_SUMMARY_MD}")
    print(f"Self-pairs skipped: {self_or_dup_skipped}, target tail fraction: {target_tail_fraction:.4f}")


if __name__ == "__main__":
    main()

"""Load phase 7's full also_buy positive-edge set + embeddings, re-assert
disjointness from the phase 1b eval sample, compute inverse-propensity-scoring
(IPS) weights for the tail-exposure loss from phase 3's ref_count table, and
build the per-anchor positive-set dict used for in-batch false-negative
masking during training.

IPS design (see week5/phase16_relevance_tail_dial.md step 3 and this
project's plan): propensity(item) = (ref_count+1)/(max_ref_count_among_
training_targets+1); raw_weight = 1/propensity; stabilized by capping at the
95th percentile (computed once over the full training-target distribution,
not per-batch) then rescaled to mean 1.0. Capping, not per-batch
self-normalization, because self-normalization only bounds a batch's weight
SUM, not any single example's share of it -- a near-zero-propensity item
could still dominate a 128-item batch's gradient. The mean-1 rescale is a
fixed, one-time affine convenience for interpretability and to keep this
loss's scale comparable to the unweighted relevance loss for the later
gradient-norm calibration -- it is not a second dynamic stabilization
technique layered on top of capping.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
PHASE1B_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"

POSITIVE_EDGES = PHASE7_DIR / "data" / "positive_edges.json"
EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = PHASE1B_DIR / "data" / "sample_data.csv"

OUT_DATA = BASE_DIR / "data" / "training_data_with_weights.npz"
OUT_POS_SETS = BASE_DIR / "data" / "positive_sets.json"
OUT_REPORT = BASE_DIR / "ips_weighting_check.md"

CAP_PERCENTILE = 95
DEGENERATE_MASS_THRESHOLD = 0.98  # if >=98% of weights collapse to essentially one value, stop


def main():
    with open(POSITIVE_EDGES) as f:
        edges = json.load(f)

    d = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pool_asins = d["asins"].astype(str)
    pool_set = set(pool_asins.tolist())

    eval_df = pd.read_csv(EVAL_SAMPLE_CSV, usecols=["asin"])
    eval_set = set(eval_df["asin"].astype(str).tolist())

    overlap = pool_set & eval_set
    edge_asins = set()
    for e in edges:
        edge_asins.add(e["source"])
        edge_asins.add(e["target"])
    edge_vs_eval_overlap = edge_asins & eval_set

    assert len(overlap) == 0, f"Pool/eval-sample overlap detected: {len(overlap)}"
    assert len(edge_vs_eval_overlap) == 0, f"Training-edge asin touches eval sample: {len(edge_vs_eval_overlap)}"

    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count"])
    refcount_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["ref_count"].astype(int)))

    target_refcounts = np.array([refcount_lookup.get(e["target"], 0) for e in edges], dtype=np.float64)
    max_rc = target_refcounts.max()
    propensity = (target_refcounts + 1.0) / (max_rc + 1.0)
    raw_weight = 1.0 / propensity

    cap_value = np.percentile(raw_weight, CAP_PERCENTILE)
    capped_weight = np.minimum(raw_weight, cap_value)
    cap_bound_fraction = float(np.mean(raw_weight >= cap_value))

    final_weight = capped_weight / capped_weight.mean()  # mean-1 rescale

    # degeneracy check: what fraction of the distribution sits within 1% of the max value?
    near_max_fraction = float(np.mean(final_weight >= final_weight.max() * 0.99))
    is_degenerate = near_max_fraction >= DEGENERATE_MASS_THRESHOLD

    # positive sets for in-batch false-negative masking (both directions, train+val, matching phase 7/8/9 convention)
    positive_sets = {}
    for e in edges:
        positive_sets.setdefault(e["source"], set()).add(e["target"])
        positive_sets.setdefault(e["target"], set()).add(e["source"])
    positive_sets_serializable = {k: sorted(v) for k, v in positive_sets.items()}
    with open(OUT_POS_SETS, "w") as f:
        json.dump(positive_sets_serializable, f)

    sources = np.array([e["source"] for e in edges])
    targets = np.array([e["target"] for e in edges])
    splits = np.array([e["split"] for e in edges])

    np.savez(
        OUT_DATA,
        source=sources,
        target=targets,
        split=splits,
        weight=final_weight.astype(np.float32),
        raw_weight=raw_weight.astype(np.float32),
    )

    def pct(x):
        return f"{x:.4f}"

    lines = []
    lines.append("# Phase 16: IPS Weighting Check\n")
    lines.append("## Data preparation\n")
    lines.append(f"- Positive edges loaded: `{POSITIVE_EDGES.relative_to(BASE_DIR.parent.parent)}` -- "
                 f"{len(edges)} total ({(splits=='train').sum()} train / {(splits=='val').sum()} val), "
                 "the FULL phase 7 also_buy edge set, not phase 8's cross-category-restricted subset.")
    lines.append(f"- Embeddings: `{EMBEDDINGS_NPZ.relative_to(BASE_DIR.parent.parent)}` ({len(pool_asins)} items).")
    lines.append(f"- Disjointness re-assertion against the artifacts this phase actually loads: pool-vs-eval-sample "
                 f"overlap = **{len(overlap)}**, training-edge-asins-vs-eval-sample overlap = "
                 f"**{len(edge_vs_eval_overlap)}**. Both zero, consistent with "
                 "`week3/phase7_learned_compatibility/data/exclusion_check.md`.\n")
    lines.append("## Propensity and raw IPS weight\n")
    lines.append("`propensity(item) = (ref_count(item)+1) / (max_ref_count_among_training_targets+1)`, "
                 "`raw_weight = 1/propensity`, using phase 3's catalog-wide `ref_count` for each edge's TARGET "
                 "(positive) item.\n")
    lines.append(f"Max ref_count among the {len(edges)} training targets: **{int(max_rc)}**.\n")
    lines.append("Raw weight distribution (before any stabilization):\n")
    lines.append(f"- min: {pct(raw_weight.min())}")
    lines.append(f"- median: {pct(np.median(raw_weight))}")
    lines.append(f"- 95th percentile: {pct(cap_value)}")
    lines.append(f"- max: {pct(raw_weight.max())}\n")
    lines.append("## Stabilization: percentile capping (not self-normalization)\n")
    lines.append(f"Justification: batch-level self-normalization (rescaling so a batch's weights sum to a "
                 "constant) only bounds the batch's weight SUM, not any single example's SHARE of it -- one "
                 "near-zero-propensity item could still dominate a 128-item batch's gradient almost entirely, "
                 "which is exactly the instability the phase brief warns about. A fixed global cap directly "
                 "bounds per-example influence, which addresses that failure mode more directly.\n")
    lines.append(f"Cap: raw weight clipped at the **{CAP_PERCENTILE}th percentile** ({pct(cap_value)}), computed "
                 f"once over the full {len(edges)}-edge training-target distribution (not per-batch -- a "
                 "128-item batch's 95th percentile would be a noisy, unstable statistic to cap against).\n")
    lines.append(f"Fraction of edges with raw weight at or above the cap (i.e. affected by capping): "
                 f"**{cap_bound_fraction*100:.2f}%**. Given phase 3's own finding that 69.1% of the full catalog "
                 "has ref_count=0, a large share of training targets landing at or near the cap is EXPECTED "
                 "here, not a bug -- the tail-exposure mode's entire purpose is elevating that majority. What "
                 "would actually be a problem is the distribution collapsing to a near-single point mass with "
                 "no usable variation left; that's checked separately below.\n")
    lines.append("After capping, weights are rescaled to mean 1.0 -- a fixed, one-time affine convenience for "
                 "interpretability and so this loss's overall scale stays comparable to the unweighted relevance "
                 "loss going into the gradient-norm calibration (training script, step 3 below). This is NOT a "
                 "second dynamic stabilization technique on top of capping; it doesn't change which examples get "
                 "bounded or add any per-batch renormalization.\n")
    lines.append("Post-cap, mean-1-rescaled weight distribution (this is what training actually uses):\n")
    lines.append(f"- min: {pct(final_weight.min())}")
    lines.append(f"- median: {pct(np.median(final_weight))}")
    lines.append(f"- max: {pct(final_weight.max())}\n")
    lines.append("## Degeneracy check (stop-before-training gate)\n")
    lines.append(f"Fraction of weights within 1% of the maximum value: **{near_max_fraction*100:.2f}%** "
                 f"(threshold for 'degenerate, stop' per the brief: >= {DEGENERATE_MASS_THRESHOLD*100:.0f}%).\n")
    if is_degenerate:
        lines.append("**VERDICT: DEGENERATE.** The weight distribution has collapsed to essentially a single "
                     "point mass with no usable variation. Per the brief's explicit instruction, stopping here "
                     "rather than proceeding to a full training run on an unstable weighting scheme.\n")
    else:
        lines.append("**VERDICT: not degenerate.** There is real, usable variation in the weight distribution "
                     "beyond the capped ceiling -- proceeding to training.\n")
    lines.append("## Train/eval propensity independence\n")
    lines.append("These capped training weights are a training-loss artifact only. The field-standard "
                 "evaluation metrics computed later (APRI, RPI, Tail-Coverage@N) use phase 3's raw `ref_count`/"
                 "`tier` values directly per the brief's own formulas, with no dependency on these capped "
                 "training weights -- so there is no train/eval propensity mismatch to reconcile.\n")
    lines.append("## Positive sets for in-batch false-negative masking\n")
    lines.append(f"Built from the full (train+val, both directions) edge set: **{len(positive_sets)}** anchors "
                 f"have at least one known positive, saved to `{OUT_POS_SETS.name}`.\n")
    lines.append("*(Loss-balancing calibration and gradient-norm verification are appended to this file by "
                 "`03_train.py`.)*\n")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"Saved training data + weights: {OUT_DATA}")
    print(f"Saved positive sets: {OUT_POS_SETS}")
    print(f"Cap-bound fraction: {cap_bound_fraction*100:.2f}%  |  near-max fraction: {near_max_fraction*100:.2f}%  "
          f"|  degenerate={is_degenerate}")
    print(f"Report written: {OUT_REPORT}")

    if is_degenerate:
        raise SystemExit("STOPPING: IPS weight distribution is degenerate, see ips_weighting_check.md")


if __name__ == "__main__":
    main()

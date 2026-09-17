"""Bonus diagnostic (not in the brief's required steps, added because it
directly tests the leading mechanistic hypothesis for why axis 1/2 reversed):
does tail-exposure mode's top-K retrieval share the query's fine-grained
category more often than relevance mode's does? If so, that's direct
evidence tail-exposure mode learned a same-category-clustering axis (an
artifact of training on same-category attribute pairs) rather than a
tail-popularity axis specifically -- which would explain the reversal
without needing to guess.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
PHASE8_DIR = BASE_DIR.parent.parent / "week3" / "phase8_cross_category_retraining"
PHASE1B_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"

PRODUCT_TYPES_JSON = PHASE8_DIR / "data" / "product_types.json"
QUERY_CSV = PHASE1B_DIR / "data" / "sample_data.csv"
CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_MD = BASE_DIR / "logs" / "category_clustering_diagnostic.md"

K = 5


def breadcrumb_type(cat_str):
    try:
        cat = ast.literal_eval(cat_str)
        if isinstance(cat, list) and len(cat) > 3:
            return cat[3]
    except (ValueError, SyntaxError):
        pass
    return None


def main():
    import json
    with open(PRODUCT_TYPES_JSON) as f:
        gallery_types = json.load(f)

    query_df = pd.read_csv(QUERY_CSV, usecols=["asin", "category"])
    query_types = {row["asin"]: breadcrumb_type(row["category"]) for _, row in query_df.iterrows()}

    d = np.load(CACHE, allow_pickle=True)
    alphas = [round(float(a), 1) for a in d["alphas"]]
    query_asins = d["query_asins"].astype(str)
    retrieved = d["retrieved_asins"][:, :, :K]

    rows = []
    for ai, alpha in enumerate(alphas):
        same_cat_fractions = []
        for qi, qa in enumerate(query_asins):
            qtype = query_types.get(qa)
            if qtype is None:
                continue
            topk = retrieved[ai, qi]
            same = sum(1 for item in topk if gallery_types.get(item) == qtype)
            same_cat_fractions.append(same / K)
        rows.append({"alpha": alpha, "mean_same_category_fraction": float(np.mean(same_cat_fractions)),
                      "n_queries_with_known_type": len(same_cat_fractions)})
        print(f"alpha={alpha:.1f}: mean_same_category_fraction@{K} = {rows[-1]['mean_same_category_fraction']:.4f}")

    r0 = next(r for r in rows if r["alpha"] == 0.0)
    r1 = next(r for r in rows if r["alpha"] == 1.0)

    lines = ["# Phase 16c: Category-Clustering Diagnostic (bonus, not a required step)\n"]
    lines.append(f"Tests the leading mechanistic hypothesis for the axis-check reversal directly: does "
                 f"tail-exposure mode's top-{K} retrieval share the query's fine-grained category "
                 "(phase 8's product_types.json breadcrumb) more often than relevance mode's does?\n")
    lines.append(f"Queries with a known fine-grained type: {r0['n_queries_with_known_type']}/{len(query_asins)}.\n")
    lines.append("| alpha | Mean same-category fraction @5 |")
    lines.append("|---|---|")
    for r in rows:
        lines.append(f"| {r['alpha']:.1f} | {r['mean_same_category_fraction']:.4f} |")
    lines.append("")
    lines.append(f"**Tail-exposure (a=0.0): {r0['mean_same_category_fraction']:.4f} vs. Relevance (a=1.0): "
                 f"{r1['mean_same_category_fraction']:.4f}**\n")
    if r0["mean_same_category_fraction"] > r1["mean_same_category_fraction"] + 0.02:
        lines.append("**Confirms the hypothesis**: tail-exposure mode retrieves same-fine-grained-category "
                     "items substantially more often than relevance mode does -- direct evidence it learned a "
                     "same-category-clustering axis (an artifact of training on same-category attribute pairs) "
                     "rather than a tail-popularity axis specifically. Since this project's own phase 6/8 "
                     "found ~74% of also_buy edges are also same-category, this clustering axis partially "
                     "overlaps with what the Hit-Rate ground truth rewards, plausibly explaining why "
                     "tail-exposure mode edged out relevance mode on Hit-Rate/ref_count rather than the "
                     "reverse.")
    else:
        lines.append("Does not clearly confirm the same-category-clustering hypothesis -- the reversal likely "
                     "has a different or additional mechanistic explanation, reported as such rather than "
                     "forcing this explanation to fit.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()

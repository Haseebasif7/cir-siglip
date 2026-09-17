"""Field-standard long-tail metrics (Coverage@N, Tail-Coverage@N, APRI, RPI),
using the exact formulas from the week 5 literature review's systematic-
review source, at N=10 and N=20 (matching GUME's own convention, useful for
the contextual comparison). Reuses the cached alpha-sweep retrievals from
04_evaluate_alpha_sweep.py rather than recomputing retrieval.

Denominators for Coverage@N/Tail-Coverage@N are the TRUE catalog-wide counts
from phase 3 (3,777,545 total items; 1,888,773 tail-tier items) -- NOT the
26,591-item reachable gallery size. The brief requires the field's own
catalog-relative definitions ("confirmed directly from the systematic
review's full text, don't approximate or invent alternative versions"), so
this will produce small absolute percentages (upper-bounded by
gallery_size/catalog_size, ~0.70%) -- an honest structural ceiling of this
pipeline's finite embedded pool, reported as such, not normalized away.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"
CATALOG_TOTAL = 3_777_545
CATALOG_TAIL_TOTAL = 1_888_773

CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_RESULTS_MD = BASE_DIR / "results_table.md"

N_VALUES = [10, 20]

GUME_CLOTHING = {  # from week5/literature_review, GUME's own reported numbers on the shared category
    "R@10": 0.0703, "R@20": 0.1024,
}


def main():
    d = np.load(CACHE, allow_pickle=True)
    alphas = d["alphas"]
    retrieved_asins = d["retrieved_asins"]  # (n_alpha, n_query, K_CACHE)

    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count", "tier"])
    refcount_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["ref_count"].astype(int)))
    tier_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["tier"]))

    rows = {N: [] for N in N_VALUES}
    for ai, alpha in enumerate(alphas):
        for N in N_VALUES:
            topN = retrieved_asins[ai, :, :N]  # (n_query, N)
            flat = topN.reshape(-1)

            unique_items = set(flat.tolist())
            coverage = len(unique_items) / CATALOG_TOTAL

            tail_items_seen = {a for a in unique_items if tier_lookup.get(a) == "tail"}
            tail_coverage = len(tail_items_seen) / CATALOG_TAIL_TOTAL

            refcounts = np.array([refcount_lookup.get(a, 0) for a in flat], dtype=np.float64)
            apri = float(refcounts.mean())  # pooled mean == macro-average of per-query means, N constant per query

            is_head = np.array([tier_lookup.get(a) == "head" for a in flat])
            rpi = float(is_head.mean())

            rows[N].append({
                "alpha": float(alpha), "coverage": coverage, "tail_coverage": tail_coverage,
                "apri": apri, "rpi": rpi, "n_unique": len(unique_items), "n_unique_tail": len(tail_items_seen),
            })
            print(f"N={N} alpha={alpha:.1f}: coverage={coverage:.6f} tail_coverage={tail_coverage:.6f} "
                  f"apri={apri:.2f} rpi={rpi:.4f}")

    lines = ["\n## Field-standard long-tail metrics\n"]
    lines.append("Formulas from the week 5 literature review (Saha, Biswas, Das & Paitya's systematic review). "
                 f"Coverage@N and Tail-Coverage@N denominators are the TRUE catalog-wide counts from phase 3 "
                 f"({CATALOG_TOTAL:,} total items, {CATALOG_TAIL_TOTAL:,} tail-tier items) -- not the "
                 "26,591-item reachable gallery. Expect small absolute Coverage/Tail-Coverage percentages as a "
                 "result; this is a structural ceiling from this pipeline's finite embedded pool, not a "
                 "computation error.\n")

    for N in N_VALUES:
        lines.append(f"### N={N}\n")
        lines.append("| alpha | Coverage@N | Tail-Coverage@N | APRI | RPI |")
        lines.append("|---|---|---|---|---|")
        for r in rows[N]:
            lines.append(f"| {r['alpha']:.1f} | {r['coverage']*100:.4f}% ({r['n_unique']}) | "
                         f"{r['tail_coverage']*100:.4f}% ({r['n_unique_tail']}) | {r['apri']:.2f} | {r['rpi']:.4f} |")
        lines.append("")

    # endpoint deltas for the honest-interpretation writeup
    lines.append("### Endpoint deltas (alpha=1.0 relevance vs alpha=0.0 tail-exposure)\n")
    for N in N_VALUES:
        r1 = next(r for r in rows[N] if r["alpha"] == 1.0)
        r0 = next(r for r in rows[N] if r["alpha"] == 0.0)
        lines.append(f"- N={N}: Coverage@N {r0['coverage']*100:.4f}% (tail) vs {r1['coverage']*100:.4f}% "
                     f"(relevance); Tail-Coverage@N {r0['tail_coverage']*100:.4f}% (tail) vs "
                     f"{r1['tail_coverage']*100:.4f}% (relevance); APRI {r0['apri']:.2f} (tail) vs "
                     f"{r1['apri']:.2f} (relevance); RPI {r0['rpi']:.4f} (tail) vs {r1['rpi']:.4f} (relevance).")
    lines.append("")

    lines.append("## Contextual comparison: GUME (CIKM 2024), Clothing/Shoes/Jewelry category\n")
    lines.append("Reported as context, not a matched-protocol baseline claim -- GUME requires a full "
                 "user-item interaction graph, this project's mechanism is item-only; the two are not the same "
                 "evaluation protocol (GUME: Recall@K on a held-out user-item interaction split; this phase: "
                 "Hit-Rate@K on an item-to-item also_buy retrieval task over a different, smaller candidate "
                 "pool). Numbers below are GUME's own published results on the same Amazon Clothing, Shoes, "
                 "and Jewelry category this project uses throughout, confirmed directly from the paper's full "
                 "text (`week5/literature_review/longtail_direction_literature_review.md`):\n")
    lines.append(f"- GUME Recall@10 = {GUME_CLOTHING['R@10']}, Recall@20 = {GUME_CLOTHING['R@20']} "
                 "(vs. MENTOR, its strongest baseline: 0.0668 / 0.0989).\n")

    with open(OUT_RESULTS_MD, "a") as f:
        f.write("\n".join(lines))
    print(f"Appended field-standard metrics + GUME comparison to {OUT_RESULTS_MD}")


if __name__ == "__main__":
    main()

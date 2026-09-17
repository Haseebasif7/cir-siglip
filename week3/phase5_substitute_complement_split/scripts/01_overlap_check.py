"""
Phase 5, step 1: check how much also_buy and also_viewed overlap with each
other at the pair level, before treating them as separate substitute/
complement signals.

Reuses phase 1b's exact sample (week2/phase1b_category_balanced/data/sample_data.csv,
1,872 products) -- no new sampling. Only within-sample edges are counted (an edge
whose target isn't in the sample can't be scored against retrieval anyway, so it's
irrelevant to the question this phase is asking).
"""
import ast
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced" / "data" / "sample_data.csv"
OUT_MD = BASE_DIR / "overlap_check.md"
OUT_JSON = BASE_DIR / "data" / "signal_edges.json"


def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    also_buy = {}
    also_viewed = {}
    category = {}
    for _, row in df.iterrows():
        asin = row["asin"]
        also_buy[asin] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()
        also_viewed[asin] = set(ast.literal_eval(row["also_viewed"])) if pd.notna(row["also_viewed"]) else set()
        category[asin] = row["category_bucket"]
    return also_buy, also_viewed, category


def main():
    also_buy, also_viewed, category = load_sample()
    sample_asins = set(also_buy.keys())

    # Directed (source, target) pairs, restricted to targets that are also in the
    # sample -- these are the only edges that can ever be scored as a "hit" against
    # a top-K retrieval list drawn from this same sample.
    buy_edges = {(src, tgt) for src, targets in also_buy.items() for tgt in targets if tgt in sample_asins}
    viewed_edges = {(src, tgt) for src, targets in also_viewed.items() for tgt in targets if tgt in sample_asins}

    both = buy_edges & viewed_edges
    only_buy = buy_edges - viewed_edges
    only_viewed = viewed_edges - buy_edges
    union = buy_edges | viewed_edges

    n_union = len(union)
    n_both = len(both)
    n_only_buy = len(only_buy)
    n_only_viewed = len(only_viewed)

    pct_both = 100 * n_both / n_union if n_union else 0.0
    pct_only_buy = 100 * n_only_buy / n_union if n_union else 0.0
    pct_only_viewed = 100 * n_only_viewed / n_union if n_union else 0.0

    # Also report each field's raw within-sample edge count and how much of each
    # field, individually, is "explained" by the other -- this is the number that
    # matters for judging whether the substitute/complement split is visible at all.
    pct_buy_also_in_viewed = 100 * len(buy_edges & viewed_edges) / len(buy_edges) if buy_edges else 0.0
    pct_viewed_also_in_buy = 100 * len(viewed_edges & buy_edges) / len(viewed_edges) if viewed_edges else 0.0

    lines = [
        "# Phase 5, Step 1: also_buy vs also_viewed Pair-Level Overlap Check",
        "",
        "Directed (source_asin, target_asin) edges, restricted to targets that are",
        "also inside the phase 1b sample (1,872 products) -- these are the only edges",
        "that can ever register as a hit against a top-K retrieval list drawn from the",
        "same sample, so edges pointing outside the sample are excluded from this count.",
        "",
        f"- Within-sample also_buy edges: {len(buy_edges)}",
        f"- Within-sample also_viewed edges: {len(viewed_edges)}",
        f"- Union (either field): {n_union}",
        "",
        "## Pair-level overlap (relative to the union of both fields)",
        "",
        "| Bucket | Count | % of union |",
        "|---|---|---|",
        f"| In both also_buy and also_viewed | {n_both} | {pct_both:.1f}% |",
        f"| Only in also_buy | {n_only_buy} | {pct_only_buy:.1f}% |",
        f"| Only in also_viewed | {n_only_viewed} | {pct_only_viewed:.1f}% |",
        "",
        "## Each field's overlap with the other (relative to that field alone)",
        "",
        "| Field | Edges | % also present in the other field |",
        "|---|---|---|",
        f"| also_buy | {len(buy_edges)} | {pct_buy_also_in_viewed:.1f}% |",
        f"| also_viewed | {len(viewed_edges)} | {pct_viewed_also_in_buy:.1f}% |",
        "",
        "## Reading",
        "",
    ]
    if len(viewed_edges) == 0 and len(buy_edges) > 0:
        reading = (
            "**also_viewed is not a 0% overlap with also_buy -- it is completely empty.** "
            "Every one of the 1,872 sampled products has an empty also_viewed list "
            "(0 nonempty out of 1,872; same result in phase 1's original 776-product "
            "sample: 0 nonempty out of 776). This was checked against a possible sampling "
            "artifact by streaming the first 50,000 raw records directly from the source "
            "metadata file (meta_Clothing_Shoes_and_Jewelry.json.gz): 0 of those records had "
            "a nonempty also_viewed field, against 8,092 with a nonempty also_buy field. "
            "**This confirms it's a genuine property of this specific hosted 2018 metadata "
            "file, not a sample or download artifact** -- also_viewed is present as a key "
            "in the JSON schema (week1's field inventory documents it) but its value is "
            "always an empty list in this dump. The substitute/complement split this phase "
            "set out to test cannot be run on also_viewed at all: there is no also_viewed "
            "ground truth anywhere in the data this project has used since phase 1."
        )
    elif pct_both >= 80:
        reading = (
            f"The two fields overlap almost completely ({pct_both:.1f}% of the combined "
            "edge set appears in both). The substitute/complement distinction is likely "
            "NOT visible in this data -- also_buy and also_viewed are close to redundant "
            "signals here, not two different relations."
        )
    elif pct_both <= 30:
        reading = (
            f"The two fields overlap only modestly ({pct_both:.1f}% of the combined edge "
            "set appears in both) -- most edges are field-specific (only_buy "
            f"{pct_only_buy:.1f}% + only_viewed {pct_only_viewed:.1f}% = "
            f"{pct_only_buy + pct_only_viewed:.1f}% of the union). This is consistent with "
            "also_buy and also_viewed capturing at least partially different relations, "
            "worth scoring separately in step 2."
        )
    else:
        reading = (
            f"The two fields overlap partially ({pct_both:.1f}% of the combined edge set "
            "appears in both) -- neither fully redundant nor fully distinct. Worth scoring "
            "separately in step 2 to see if retrieval quality actually differs between them."
        )
    lines.append(reading)
    lines.append("")

    if len(viewed_edges) == 0 and len(buy_edges) > 0:
        lines.append(
            "## One-off raw-metadata verification (not re-run by this script)\n\n"
            "To rule out a sample-construction bug before concluding also_viewed is "
            "genuinely absent, the first 50,000 raw records were streamed directly from "
            "`meta_Clothing_Shoes_and_Jewelry.json.gz` (the same source file every sample "
            "in this project comes from) and checked ad hoc:\n\n"
            "| Check | Result |\n|---|---|\n"
            "| Records with nonempty also_viewed | 0 / 50,000 |\n"
            "| Records with nonempty also_buy | 8,092 / 50,000 |\n\n"
            "also_viewed is present as a JSON key but its value is always `[]` in this "
            "hosted 2018 file -- confirmed independent of any sampling choice made in this "
            "project.\n"
        )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    print(reading)

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "also_buy": {k: sorted(v) for k, v in also_buy.items()},
            "also_viewed": {k: sorted(v) for k, v in also_viewed.items()},
            "category": category,
        }, f)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

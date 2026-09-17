"""Step 1: filter phase 7's full also_buy edge set down to edges whose
TARGET item is tail-tier (phase 3's tier definition), the new tail-exposure
training signal for this phase -- restriction, not reweighting. Same
discipline phase 8 used when it checked its own filtered edge count before
training: report the actual count, and stop/flag rather than proceed on a
thin signal if it's too small to train meaningfully.
"""
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"
POSITIVE_EDGES = PHASE7_DIR / "data" / "positive_edges.json"

OUT_EDGES = BASE_DIR / "data" / "tail_tier_edges.json"
OUT_REPORT = BASE_DIR / "edge_filtering_summary.md"

# Same discipline as phase 8's own edge-count check: a bounded minimum before
# training is considered meaningful rather than a thin signal.
MIN_TRAIN_EDGES = 1000


def main():
    with open(POSITIVE_EDGES) as f:
        edges = json.load(f)

    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "tier"])
    tier_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["tier"]))

    tail_edges = [e for e in edges if tier_lookup.get(e["target"]) == "tail"]
    head_edges = [e for e in edges if tier_lookup.get(e["target"]) == "head"]
    mid_edges = [e for e in edges if tier_lookup.get(e["target"]) == "mid"]
    unknown_edges = [e for e in edges if tier_lookup.get(e["target"]) is None]

    train_tail = [e for e in tail_edges if e["split"] == "train"]
    val_tail = [e for e in tail_edges if e["split"] == "val"]

    with open(OUT_EDGES, "w") as f:
        json.dump(tail_edges, f)

    # unique anchors and unique targets in the tail-restricted set, worth reporting
    # since a small number of unique targets (even with many edges) would itself be a thin signal
    unique_anchors = {e["source"] for e in tail_edges}
    unique_targets = {e["target"] for e in tail_edges}

    sufficient = len(train_tail) >= MIN_TRAIN_EDGES

    lines = [
        "# Phase 16b, Step 1: Tail-Tier Edge Filtering\n",
        f"Source: `{POSITIVE_EDGES.relative_to(BASE_DIR.parent.parent)}` -- {len(edges)} total also_buy edges "
        "(phase 7's full, unrestricted set, same one phase 16's relevance mode trains on).\n",
        "## Tier breakdown of edge TARGETS (the restriction variable)\n",
        "| Tier | Edge count | % of total |",
        "|---|---|---|",
        f"| head | {len(head_edges)} | {len(head_edges)/len(edges)*100:.2f}% |",
        f"| mid | {len(mid_edges)} | {len(mid_edges)/len(edges)*100:.2f}% |",
        f"| tail | {len(tail_edges)} | {len(tail_edges)/len(edges)*100:.2f}% |",
        f"| unknown (not in phase 3's tier lookup) | {len(unknown_edges)} | {len(unknown_edges)/len(edges)*100:.2f}% |",
        "",
        "## Tail-tier-restricted edge set (this phase's tail-exposure training signal)\n",
        f"- Total tail-tier edges: **{len(tail_edges)}**",
        f"- Train split: **{len(train_tail)}**",
        f"- Val split: **{len(val_tail)}**",
        f"- Unique anchor (source) items: **{len(unique_anchors)}**",
        f"- Unique tail-tier target items: **{len(unique_targets)}**",
        "",
        f"Minimum bar for 'meaningful to train on' (this phase's own pre-declared threshold): "
        f"{MIN_TRAIN_EDGES} train edges.\n",
    ]
    if sufficient:
        lines.append(f"**VERDICT: SUFFICIENT.** {len(train_tail)} train edges clears the "
                     f"{MIN_TRAIN_EDGES}-edge bar -- proceeding to step 2 (training).")
    else:
        lines.append(f"**VERDICT: INSUFFICIENT.** {len(train_tail)} train edges falls short of the "
                     f"{MIN_TRAIN_EDGES}-edge bar -- per the brief's explicit instruction, stopping here and "
                     "reporting this directly rather than training on a thin signal.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"Tail-tier edges: {len(tail_edges)} total ({len(train_tail)} train / {len(val_tail)} val)")
    print(f"Unique anchors: {len(unique_anchors)}, unique targets: {len(unique_targets)}")
    print(f"Sufficient: {sufficient}")
    print(f"Saved: {OUT_EDGES}")
    print(f"Report: {OUT_REPORT}")

    if not sufficient:
        raise SystemExit("STOPPING: tail-tier edge count is below the meaningful-training threshold.")


if __name__ == "__main__":
    main()

"""Step 2: verify the attribute-based pairs are genuinely different from
the also_buy edges used in phases 16 and 16b, not a disguised repeat --
checked directly, not assumed.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"

ALSO_BUY_EDGES = PHASE7_DIR / "data" / "positive_edges.json"
ATTRIBUTE_EDGES = BASE_DIR / "data" / "attribute_pairs.json"
OUT_REPORT = BASE_DIR / "attribute_pairs_summary.md"


def main():
    with open(ALSO_BUY_EDGES) as f:
        also_buy = json.load(f)
    with open(ATTRIBUTE_EDGES) as f:
        attribute = json.load(f)

    also_buy_pairs = {(e["source"], e["target"]) for e in also_buy}
    attribute_pairs = {(e["source"], e["target"]) for e in attribute}

    overlap = also_buy_pairs & attribute_pairs
    overlap_frac_of_attribute = len(overlap) / len(attribute_pairs) if attribute_pairs else 0.0
    overlap_frac_of_also_buy = len(overlap) / len(also_buy_pairs) if also_buy_pairs else 0.0

    # also check undirected overlap (a pair counted as "the same edge" regardless of direction)
    also_buy_undirected = {frozenset(p) for p in also_buy_pairs}
    attribute_undirected = {frozenset(p) for p in attribute_pairs}
    overlap_undirected = also_buy_undirected & attribute_undirected
    overlap_undirected_frac = len(overlap_undirected) / len(attribute_undirected) if attribute_undirected else 0.0

    lines = ["\n## Step 2: overlap check against phase 16/16b's also_buy edges\n"]
    lines.append(f"- Also_buy directed edges (phase 16/16b's signal): {len(also_buy_pairs)}")
    lines.append(f"- Attribute-based directed edges (this phase's signal): {len(attribute_pairs)}")
    lines.append(f"- Directed overlap (exact same (source, target) pair in both): **{len(overlap)}** "
                 f"({overlap_frac_of_attribute*100:.3f}% of attribute pairs, "
                 f"{overlap_frac_of_also_buy*100:.3f}% of also_buy pairs)")
    lines.append(f"- Undirected overlap (same two items paired, either direction): **{len(overlap_undirected)}** "
                 f"({overlap_undirected_frac*100:.3f}% of attribute pairs)")
    lines.append("")
    if overlap_frac_of_attribute < 0.05:
        lines.append(f"**Verdict: genuinely different signal.** Overlap is negligible "
                     f"({overlap_frac_of_attribute*100:.3f}% of attribute pairs also appear as also_buy edges) "
                     "-- this is not a disguised repeat of phases 16/16b's behavioral signal, it's drawing "
                     "positive pairs from a structurally different source (shared fine-grained category, not "
                     "co-purchase history).")
    else:
        lines.append(f"**Verdict: meaningful overlap found ({overlap_frac_of_attribute*100:.1f}% of attribute "
                     "pairs also appear as also_buy edges) -- flagging this honestly rather than glossing over "
                     "it, per the brief's explicit instruction. This would mean the attribute signal is not as "
                     "structurally distinct from the failed phase 16/16b signal as intended.")
    lines.append("")

    with open(OUT_REPORT, "a") as f:
        f.write("\n".join(lines))

    print(f"Directed overlap: {len(overlap)} ({overlap_frac_of_attribute*100:.3f}% of attribute pairs)")
    print(f"Undirected overlap: {len(overlap_undirected)} ({overlap_undirected_frac*100:.3f}% of attribute pairs)")
    print(f"Appended to {OUT_REPORT}")


if __name__ == "__main__":
    main()

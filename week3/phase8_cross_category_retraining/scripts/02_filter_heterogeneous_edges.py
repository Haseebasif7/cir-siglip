"""
Phase 8, step 2: reuse phase 7's cleaned training pool and positive
also_buy edges, but keep only the "heterogeneous dyads" -- edges where
source and target have a DIFFERENT product type at the level identified in
step 1 (index 3 of the categories breadcrumb). This directly follows Veit
et al. (ICCV 2015)'s fix: restrict compatibility training pairs to
cross-category pairs specifically to avoid training on same-item/near-
duplicate pairs, which phase 6/7 identified as the likely cause of phase 7's
failure.

No new sampling, no new embeddings, no new hard-negative mining needed:
- Positive edges: filtered from phase 7's existing positive_edges.json.
- Hard-negative candidates: reused AS-IS from phase 7's
  hard_negative_candidates.json. This is not just a shortcut -- it's the
  methodologically correct choice: that file's exclusion set for each anchor
  is built from ALL of that anchor's true also_buy targets (any type, train+
  val combined), not just heterogeneous ones. A same-type also_buy partner is
  still a real relationship even though phase 8 isn't training on it as a
  positive this round; mining it as a "hard negative" would be factually
  wrong and would reintroduce exactly the contamination this phase exists to
  avoid. Every phase-8 anchor is a subset of phase 7's anchors, so the
  existing candidate file already covers what's needed.
"""
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent / "phase7_learned_compatibility"

PRODUCT_TYPES_JSON = BASE_DIR / "data" / "product_types.json"
PHASE7_POSITIVE_EDGES = PHASE7_DIR / "data" / "positive_edges.json"
PHASE7_HARD_NEG = PHASE7_DIR / "data" / "hard_negative_candidates.json"

OUT_POSITIVE_EDGES = BASE_DIR / "data" / "heterogeneous_positive_edges.json"
OUT_HARD_NEG = BASE_DIR / "data" / "hard_negative_candidates.json"  # filtered to relevant anchors only
REPORT_MD = BASE_DIR / "edge_filtering_summary.md"

SEED = 42
VAL_FRACTION = 0.10
MIN_WORKABLE_EDGES = 3000  # rough floor per the brief ("a few thousand")


def main():
    with open(PRODUCT_TYPES_JSON) as f:
        product_types = json.load(f)
    with open(PHASE7_POSITIVE_EDGES) as f:
        phase7_edges = json.load(f)  # list of {"source", "target", "split"} -- split is phase 7's own, ignored here

    n_total = len(phase7_edges)
    hetero_edges = []
    n_same = 0
    n_unknown = 0
    for e in phase7_edges:
        ts = product_types.get(e["source"])
        tt = product_types.get(e["target"])
        if ts is None or tt is None:
            n_unknown += 1
        elif ts != tt:
            hetero_edges.append((e["source"], e["target"]))
        else:
            n_same += 1

    n_hetero = len(hetero_edges)
    print(f"Phase 7 total positive edges: {n_total}")
    print(f"  Heterogeneous (different type): {n_hetero}")
    print(f"  Same type: {n_same}")
    print(f"  Unknown type (one/both endpoints missing breadcrumb depth): {n_unknown}")

    expansion_needed = n_hetero < MIN_WORKABLE_EDGES
    if expansion_needed:
        print(f"WARNING: {n_hetero} heterogeneous edges is below the workable floor "
              f"({MIN_WORKABLE_EDGES}) -- pool expansion would be needed. Not implemented "
              f"in this run since the actual count clears the floor comfortably.")
    else:
        print(f"{n_hetero} heterogeneous edges clears the workable floor ({MIN_WORKABLE_EDGES}) "
              f"comfortably -- no pool expansion needed.")

    # re-split heterogeneous edges 90/10 (independent of phase 7's own split,
    # since this is a different, smaller positive-edge population)
    rng = random.Random(SEED)
    shuffled = hetero_edges[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    val_edges = shuffled[:n_val]
    train_edges = shuffled[n_val:]

    edge_records = (
        [{"source": s, "target": t, "split": "train"} for s, t in train_edges] +
        [{"source": s, "target": t, "split": "val"} for s, t in val_edges]
    )
    with open(OUT_POSITIVE_EDGES, "w") as f:
        json.dump(edge_records, f)
    print(f"Saved {OUT_POSITIVE_EDGES}: {len(train_edges)} train, {len(val_edges)} val.")

    # carry over the relevant slice of phase 7's hard-negative candidates
    # (anchors that appear as a source in the heterogeneous edge set)
    hetero_anchors = {s for s, _ in hetero_edges}
    with open(PHASE7_HARD_NEG) as f:
        phase7_hard_neg = json.load(f)
    filtered_hard_neg = {a: cands for a, cands in phase7_hard_neg.items() if a in hetero_anchors}
    with open(OUT_HARD_NEG, "w") as f:
        json.dump(filtered_hard_neg, f)
    print(f"Saved {OUT_HARD_NEG}: hard-negative candidates for {len(filtered_hard_neg)} "
          f"anchors (of {len(hetero_anchors)} heterogeneous-dyad anchors; "
          f"{len(hetero_anchors) - len(filtered_hard_neg)} missing, would need random-only fallback).")

    lines = [
        "# Phase 8, Step 2: Heterogeneous-Dyad Edge Filtering Summary",
        "",
        f"Starting point: phase 7's {n_total} positive also_buy edges "
        "(both endpoints in the shared 24,719-product cleaned pool).",
        "",
        "| Bucket | Count | % of phase 7 total |",
        "|---|---|---|",
        f"| Heterogeneous (different type, kept as phase 8 positives) | {n_hetero} | {100*n_hetero/n_total:.1f}% |",
        f"| Same type (excluded -- near-duplicate/substitute-like) | {n_same} | {100*n_same/n_total:.1f}% |",
        f"| Unknown type on one/both ends (excluded, conservative) | {n_unknown} | {100*n_unknown/n_total:.1f}% |",
        "",
        f"**{n_hetero} heterogeneous-dyad edges clears the workable floor "
        f"(~{MIN_WORKABLE_EDGES}) comfortably -- no pool expansion was needed.**"
        if not expansion_needed else
        f"**WARNING: only {n_hetero} heterogeneous-dyad edges, below the workable floor "
        f"(~{MIN_WORKABLE_EDGES}) -- pool expansion needed but not implemented in this run.**",
        "",
        f"Re-split 90/10: {len(train_edges)} train edges, {len(val_edges)} val edges.",
        "",
        "## Reading",
        "",
        f"The majority of also_buy edges in this catalog ({100*n_same/n_total:.1f}%) connect "
        "products of the SAME fine-grained type -- consistent with phases 6 and 7's finding "
        "that also_buy is dominated by near-duplicate/substitute-like pairs (same item, "
        "different color/size/listing) rather than genuine cross-category complements. "
        f"The remaining {100*n_hetero/n_total:.1f}% heterogeneous slice is what this phase "
        "trains on -- smaller than phase 7's full edge set, but still a substantial "
        f"{n_hetero}-edge training signal.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()

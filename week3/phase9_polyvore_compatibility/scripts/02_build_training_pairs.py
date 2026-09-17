"""
Phase 9, step 2: build positive "compatible" pairs from real outfits.

For every outfit in the official nondisjoint train/valid splits, every pair
of items that appear together in that outfit is a positive pair. Unlike
Amazon's also_buy edges (inherently directed, purchase-order-like),
co-outfit membership is symmetric -- both (A,B) and (B,A) are generated as
separate directed training edges, so both items get to serve as anchor
during training (this project's training loop needs a directed
anchor->positive framing for the MNRL loss regardless of whether the
underlying relation is symmetric).

Uses the OFFICIAL train.json / valid.json split directly for train/val,
rather than re-deriving a random split like phases 7/8 did for Amazon --
Polyvore already provides an outfit-level official split, and reusing it
avoids any risk of the same outfit's pairs leaking across train/val (a risk
phase 7/8 didn't face since Amazon has no "outfit" grouping above the edge
level).

Checks the brief's own stated assumption (most pairs should already be
cross-category, since outfits are typically one item per role) against the
actual semantic_category labels from step 1, rather than assuming it holds.
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "polyvore_raw"
IMAGES_DIR = BASE_DIR / "data" / "images"

POSITIVE_EDGES_JSON = BASE_DIR / "data" / "positive_edges.json"
REPORT_MD = BASE_DIR / "edge_summary.md"


def load_item_types():
    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    return {item_id: v.get("semantic_category") for item_id, v in meta.items()}


def load_outfits(split):
    with open(RAW_DIR / "nondisjoint" / f"{split}.json") as f:
        return json.load(f)


def build_pairs(outfits, valid_items):
    edges = []
    n_skipped_missing_image = 0
    for outfit in outfits:
        item_ids = [it["item_id"] for it in outfit["items"]]
        item_ids = [i for i in item_ids if i in valid_items]
        n_skipped_missing_image += len(outfit["items"]) - len(item_ids)
        for a, b in combinations(item_ids, 2):
            edges.append((a, b))
            edges.append((b, a))  # symmetric: co-outfit membership has no direction
    return edges, n_skipped_missing_image


def main():
    types = load_item_types()
    valid_items = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
    print(f"{len(valid_items)} items have a downloaded image.")

    train_outfits = load_outfits("train")
    valid_outfits = load_outfits("valid")

    train_edges, n_skip_train = build_pairs(train_outfits, valid_items)
    val_edges, n_skip_val = build_pairs(valid_outfits, valid_items)
    print(f"Train: {len(train_edges)} directed positive edges from {len(train_outfits)} outfits "
          f"({n_skip_train} item-slots skipped, missing image).")
    print(f"Val: {len(val_edges)} directed positive edges from {len(valid_outfits)} outfits "
          f"({n_skip_val} item-slots skipped, missing image).")

    # cross-category check, per the brief's own instruction to verify rather than assume
    def cross_category_ratio(edges):
        n_cross, n_same, n_unknown = 0, 0, 0
        for a, b in edges:
            ta, tb = types.get(a), types.get(b)
            if ta is None or tb is None:
                n_unknown += 1
            elif ta != tb:
                n_cross += 1
            else:
                n_same += 1
        return n_cross, n_same, n_unknown

    n_cross, n_same, n_unknown = cross_category_ratio(train_edges)
    total = n_cross + n_same + n_unknown
    print(f"Train edges: {n_cross} cross-category ({100*n_cross/total:.1f}%), "
          f"{n_same} same-category ({100*n_same/total:.1f}%), "
          f"{n_unknown} unknown type ({100*n_unknown/total:.1f}%)")

    edge_records = (
        [{"source": a, "target": b, "split": "train"} for a, b in train_edges] +
        [{"source": a, "target": b, "split": "val"} for a, b in val_edges]
    )
    with open(POSITIVE_EDGES_JSON, "w") as f:
        json.dump(edge_records, f)
    print(f"Saved {POSITIVE_EDGES_JSON}")

    lines = [
        "# Phase 9, Step 2: Positive Edge Summary",
        "",
        f"- Train: {len(train_edges)} directed positive edges from {len(train_outfits)} outfits "
        f"({n_skip_train} item-slots skipped, missing image)",
        f"- Val: {len(val_edges)} directed positive edges from {len(valid_outfits)} outfits "
        f"({n_skip_val} item-slots skipped, missing image)",
        "",
        "## Cross-category check (train edges), verified against actual semantic_category labels",
        "",
        "The brief's assumption: since outfits are typically assembled one item per role "
        "(a top, a bottom, shoes, etc.), most also_buy-style pairs here should already be "
        "cross-category, unlike Amazon's data -- checked directly rather than assumed:",
        "",
        "| Bucket | Count | % |",
        "|---|---|---|",
        f"| Cross-category (different semantic_category) | {n_cross} | {100*n_cross/total:.1f}% |",
        f"| Same-category (same semantic_category) | {n_same} | {100*n_same/total:.1f}% |",
        f"| Unknown type on one/both ends | {n_unknown} | {100*n_unknown/total:.1f}% |",
        "",
    ]
    if n_cross / total > 0.85:
        lines.append(f"**Confirmed: the assumption holds strongly** ({100*n_cross/total:.1f}% "
                      "cross-category) -- unlike Amazon's also_buy edges (74.1% same-category, "
                      "phase 8), Polyvore's co-outfit pairs are overwhelmingly already "
                      "cross-category by construction. No category-level correction (like "
                      "phase 8's heterogeneous-dyad filtering) is needed here -- the raw "
                      "co-outfit pairs ARE the heterogeneous-dyad-equivalent signal already.")
    else:
        lines.append(f"**Assumption only partially holds** ({100*n_cross/total:.1f}% cross-category) "
                      "-- same-category pairs are common enough here that a phase-8-style "
                      "category filter may still be worth considering, contrary to the brief's "
                      "expectation. Reporting this plainly rather than forcing the assumption.")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()

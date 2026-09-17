"""
Phase 12, step 1: build the project's own Complementary Item Retrieval (CIR)
benchmark -- see ../cir_protocol_notes.md for why this is built directly
rather than reusing github.com/bigohofone/outfit-transformer's evaluation
code (that repo's "CIR" script turned out to be 4-way FITB accuracy, not a
candidate-pool retrieval task; no usable reference implementation was found).

Reuses phase 9's already-downloaded Polyvore test split and item metadata --
no re-download. Only test.json + polyvore_item_metadata.json + the set of
downloaded image files are needed here; embeddings are loaded separately by
each evaluation script.

Output: data/cir_benchmark.json --
  {
    "pools": {category: [item_id, ...]},           # fixed background pool per category
    "queries": [
       {"outfit_id": set_id, "target_item": item_id, "category": semantic_category,
        "query_items": [item_id, ...]},             # the other items in the outfit
       ...
    ]
  }
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
IMAGES_DIR = PHASE9_DIR / "data" / "images"

OUT_JSON = BASE_DIR / "data" / "cir_benchmark.json"
REPORT_MD = BASE_DIR / "data" / "cir_benchmark_coverage.md"

POOL_CAP = 3000
SEED = 42


def load_item_types():
    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    return {item_id: v.get("semantic_category") for item_id, v in meta.items()}


def main():
    types = load_item_types()
    valid_images = {p.stem for p in IMAGES_DIR.glob("*.jpg") if not p.stem.startswith("._")}

    with open(RAW_DIR / "nondisjoint" / "test.json") as f:
        test_outfits = json.load(f)

    # A test item "qualifies" if it has a downloaded image AND a known semantic_category.
    def qualifies(item_id):
        return item_id in valid_images and types.get(item_id) is not None

    test_items_by_category = defaultdict(set)
    all_test_items = set()
    for outfit in test_outfits:
        for it in outfit["items"]:
            iid = it["item_id"]
            all_test_items.add(iid)
            if qualifies(iid):
                test_items_by_category[types[iid]].add(iid)

    n_unqualified = sum(1 for iid in all_test_items if not qualifies(iid))

    # Step 1: fixed candidate pool per category, built independent of any
    # specific query (see cir_protocol_notes.md point 2 -- this is the
    # leakage-safety property this implementation guarantees directly).
    # Capped categories keep a random 3,000-item sample; a query is only kept
    # in the benchmark if its own target happens to fall inside that sample --
    # NOT force-added afterward. An earlier version of this script force-added
    # every query's target into its pool, which defeated the cap entirely
    # (nearly every item in a popular category is *someone's* leave-one-out
    # target, so force-inclusion ballooned pools back to full-category size,
    # e.g. shoes: 3,000 -> 8,551). Restricting to queries whose target already
    # landed in the fixed sample keeps pools genuinely capped and avoids any
    # per-query pool customization, at the cost of evaluating a random subset
    # of queries for categories above the cap (reported below).
    rng = random.Random(SEED)
    pools = {}
    for cat, items in test_items_by_category.items():
        items = sorted(items)  # sort first for reproducibility before sampling
        if len(items) <= POOL_CAP:
            pools[cat] = items
        else:
            pools[cat] = sorted(rng.sample(items, POOL_CAP))
    pool_membership = {cat: set(items) for cat, items in pools.items()}

    # Step 2: leave-one-out queries over every qualifying item slot in every test
    # outfit, kept only if the target landed in its category's fixed pool.
    queries = []
    n_outfits_skipped = 0
    n_target_outside_pool = 0
    for outfit in test_outfits:
        item_ids = [it["item_id"] for it in outfit["items"]]
        qualifying = [i for i in item_ids if qualifies(i)]
        if len(qualifying) < 2:
            n_outfits_skipped += 1
            continue
        for target in qualifying:
            if target not in pool_membership[types[target]]:
                n_target_outside_pool += 1
                continue
            query_items = [i for i in qualifying if i != target]
            queries.append({
                "outfit_id": outfit["set_id"],
                "target_item": target,
                "category": types[target],
                "query_items": query_items,
            })

    with open(OUT_JSON, "w") as f:
        json.dump({"pools": pools, "queries": queries}, f)
    print(f"Saved {OUT_JSON}: {len(queries)} queries, "
          f"{sum(len(v) for v in pools.values())} total pool slots across {len(pools)} categories.")

    cat_counts = Counter(q["category"] for q in queries)
    cat_totals = Counter()
    for outfit in test_outfits:
        for it in outfit["items"]:
            if qualifies(it["item_id"]):
                cat_totals[types[it["item_id"]]] += 1
    lines = [
        "# Phase 12, Step 1: CIR Benchmark Coverage",
        "",
        f"- {len(test_outfits)} test outfits total, {n_outfits_skipped} skipped "
        "(fewer than 2 qualifying items -- none found in practice).",
        f"- {len(all_test_items)} unique test items, {n_unqualified} failed the "
        "qualifying check (missing image or unknown semantic_category).",
        f"- {n_target_outside_pool} leave-one-out item-slots dropped because their "
        "target fell outside their category's fixed 3,000-item sample (only affects "
        "categories with more than 3,000 unique test items -- see per-category table).",
        f"- **{len(queries)} leave-one-out queries kept in the final benchmark.**",
        "",
        "## Candidate pool size per category (capped at 3,000, queries whose target "
        "missed the sample are dropped rather than force-added back in)",
        "",
        "| Category | Item-slots available | Queries kept | Pool size |",
        "|---|---|---|---|",
    ]
    for cat in sorted(pools, key=lambda c: -len(pools[c])):
        lines.append(f"| {cat} | {cat_totals.get(cat, 0)} | {cat_counts.get(cat, 0)} | {len(pools[cat])} |")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()

"""
Phase 23, step 1: build a validation-split CIR benchmark, using the exact
same construction method already validated for the test benchmark
(week4/phase12_controllable_modes/scripts/01_build_cir_benchmark.py --
candidate pools capped at 3,000/category, leave-one-out queries), just
pointed at Polyvore's official `valid.json` instead of `test.json`. Kept
completely separate from the test benchmark (own file, own directory) --
every tuning decision in phase 23 reads only this file; the test benchmark
is touched exactly once, at the very end (step 7).
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
IMAGES_DIR = PHASE9_DIR / "data" / "images"

TEST_BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"
TEST_COVERAGE_MD = PHASE12_DIR / "data" / "cir_benchmark_coverage.md"

OUT_JSON = BASE_DIR / "data" / "cir_val_benchmark.json"
REPORT_MD = BASE_DIR / "validation_benchmark_construction.md"

POOL_CAP = 3000
SEED = 42


def load_item_types():
    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    return {item_id: v.get("semantic_category") for item_id, v in meta.items()}


def build_benchmark(split_name):
    """Identical logic to phase 12's 01_build_cir_benchmark.py, parametrized
    by split name instead of hardcoded to 'test'."""
    types = load_item_types()
    valid_images = {p.stem for p in IMAGES_DIR.glob("*.jpg") if not p.stem.startswith("._")}

    with open(RAW_DIR / "nondisjoint" / f"{split_name}.json") as f:
        outfits = json.load(f)

    def qualifies(item_id):
        return item_id in valid_images and types.get(item_id) is not None

    items_by_category = defaultdict(set)
    all_items = set()
    for outfit in outfits:
        for it in outfit["items"]:
            iid = it["item_id"]
            all_items.add(iid)
            if qualifies(iid):
                items_by_category[types[iid]].add(iid)

    n_unqualified = sum(1 for iid in all_items if not qualifies(iid))

    rng = random.Random(SEED)
    pools = {}
    for cat, items in items_by_category.items():
        items = sorted(items)
        if len(items) <= POOL_CAP:
            pools[cat] = items
        else:
            pools[cat] = sorted(rng.sample(items, POOL_CAP))
    pool_membership = {cat: set(items) for cat, items in pools.items()}

    queries = []
    n_outfits_skipped = 0
    n_target_outside_pool = 0
    for outfit in outfits:
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

    stats = {
        "n_outfits": len(outfits),
        "n_outfits_skipped": n_outfits_skipped,
        "n_unique_items": len(all_items),
        "n_unqualified": n_unqualified,
        "n_target_outside_pool": n_target_outside_pool,
        "n_queries": len(queries),
        "n_pool_slots": sum(len(v) for v in pools.values()),
        "n_categories": len(pools),
        "cat_pool_sizes": {c: len(v) for c, v in pools.items()},
        "cat_query_counts": dict(Counter(q["category"] for q in queries)),
    }
    return pools, queries, stats


def main():
    pools, queries, val_stats = build_benchmark("valid")

    with open(OUT_JSON, "w") as f:
        json.dump({"pools": pools, "queries": queries}, f)
    print(f"Saved {OUT_JSON}: {val_stats['n_queries']} queries, {val_stats['n_pool_slots']} pool slots.")

    # Load the test benchmark's own stats for a direct side-by-side comparison
    # (per the brief's own instruction: report directly if composition
    # differs meaningfully, don't assume the construction transfers cleanly).
    with open(TEST_BENCHMARK_JSON) as f:
        test_bench = json.load(f)
    test_pools, test_queries = test_bench["pools"], test_bench["queries"]
    test_stats = {
        "n_queries": len(test_queries),
        "n_pool_slots": sum(len(v) for v in test_pools.values()),
        "n_categories": len(test_pools),
        "cat_pool_sizes": {c: len(v) for c, v in test_pools.items()},
        "cat_query_counts": dict(Counter(q["category"] for q in test_queries)),
    }

    lines = [
        "# Phase 23, Step 1: Validation-Split CIR Benchmark Construction",
        "",
        "Built with the exact same method as the test benchmark "
        "(`week4/phase12_controllable_modes/scripts/01_build_cir_benchmark.py`: "
        "candidate pools capped at 3,000 items/category via seed=42 sampling, "
        "leave-one-out queries kept only if their target landed inside the "
        "capped sample) -- the only change is the source split file, "
        "`polyvore_raw/nondisjoint/valid.json` instead of `test.json`. Kept in "
        "its own file (`data/cir_val_benchmark.json`), never merged with or "
        "substituted for the test benchmark.",
        "",
        "## Coverage, validation split\n",
        f"- {val_stats['n_outfits']} valid-split outfits total, "
        f"{val_stats['n_outfits_skipped']} skipped (fewer than 2 qualifying items).",
        f"- {val_stats['n_unique_items']} unique valid-split items, "
        f"{val_stats['n_unqualified']} failed the qualifying check (missing image or "
        "unknown semantic_category).",
        f"- {val_stats['n_target_outside_pool']} leave-one-out item-slots dropped "
        "(target fell outside its category's capped pool sample).",
        f"- **{val_stats['n_queries']} leave-one-out queries kept**, across "
        f"{val_stats['n_categories']} category pools, {val_stats['n_pool_slots']} total pool slots.",
        "",
        "## Side-by-side against the test benchmark (already-validated construction)\n",
        "| Metric | Validation benchmark | Test benchmark |",
        "|---|---|---|",
        f"| Queries | {val_stats['n_queries']} | {test_stats['n_queries']} |",
        f"| Categories | {val_stats['n_categories']} | {test_stats['n_categories']} |",
        f"| Total pool slots | {val_stats['n_pool_slots']} | {test_stats['n_pool_slots']} |",
        "",
        "## Per-category pool size and query count, both benchmarks\n",
        "| Category | Val pool size | Val queries | Test pool size | Test queries |",
        "|---|---|---|---|---|",
    ]
    all_cats = sorted(set(val_stats["cat_pool_sizes"]) | set(test_stats["cat_pool_sizes"]),
                       key=lambda c: -test_stats["cat_pool_sizes"].get(c, 0))
    for cat in all_cats:
        lines.append(
            f"| {cat} | {val_stats['cat_pool_sizes'].get(cat, 0)} | "
            f"{val_stats['cat_query_counts'].get(cat, 0)} | "
            f"{test_stats['cat_pool_sizes'].get(cat, 0)} | "
            f"{test_stats['cat_query_counts'].get(cat, 0)} |"
        )
    lines.append("")

    # Composition check: ratio of val to test size, category-by-category
    ratios = []
    for cat in all_cats:
        v, t = val_stats["cat_pool_sizes"].get(cat, 0), test_stats["cat_pool_sizes"].get(cat, 0)
        if t > 0:
            ratios.append(v / t)
    overall_ratio = val_stats["n_queries"] / test_stats["n_queries"]
    lines.append("## Composition check\n")
    lines.append(
        f"Validation benchmark is {100*overall_ratio:.1f}% the size of the test benchmark "
        f"by query count ({val_stats['n_queries']} vs {test_stats['n_queries']}), roughly "
        f"in line with valid.json holding half as many outfits as test.json (5,000 vs "
        f"10,000). Per-category pool-size ratios range "
        f"{min(ratios)*100:.1f}%-{max(ratios)*100:.1f}% of the test benchmark's own pool "
        "sizes."
    )
    if max(ratios) / max(min(ratios), 1e-9) < 3.0:
        lines.append(
            "\n**Composition is proportionally consistent across categories** -- no category "
            "is disproportionately shrunk or missing relative to the others, so the validation "
            "benchmark's category mix is a reasonable, representative stand-in for the test "
            "benchmark's own mix, just built from fewer outfits overall."
        )
    else:
        lines.append(
            "\n**Composition is NOT proportionally consistent across categories** -- some "
            "categories shrink much more than others relative to the test benchmark. This is "
            "reported directly per the brief's own instruction, and should be kept in mind "
            "when trusting validation-benchmark comparisons as a stand-in for test-benchmark "
            "performance."
        )
    lines.append("")

    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()

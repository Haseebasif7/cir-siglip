"""
Phase 25, prerequisite: build a train-split CIR benchmark, same construction
method as the validation/test benchmarks (phase 12/23's own
01_build_cir_benchmark.py: candidate pools capped at 3,000/category,
leave-one-out queries), pointed at Polyvore's official `train.json` instead
of `valid.json`/`test.json`. Needed for step 4's overfitting check (the
brief requires comparing TRAIN and validation Recall@10 trends, not just
final validation loss/recall) -- a genuine train-side Recall@10 requires a
CIR-style benchmark built from train outfits, not just the training loss
curve, which this project has repeatedly found doesn't predict retrieval
quality.

train.json has 53,306 outfits (vs valid's 5,000), producing a much larger
query set than the val/test benchmarks if built the same way -- since this
benchmark is only used for a periodic per-epoch diagnostic (not a report
number), it's subsampled down to a fixed 5,000-query sample (seed=42) after
construction, to keep the periodic eval cost small relative to the full
22,595-query validation benchmark it's compared against every epoch.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
IMAGES_DIR = PHASE9_DIR / "data" / "images"

OUT_JSON = BASE_DIR / "data" / "cir_train_benchmark.json"
REPORT_MD = BASE_DIR / "train_benchmark_construction.md"

POOL_CAP = 3000
SEED = 42
SUBSAMPLE_QUERIES = 5000


def load_item_types():
    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    return {item_id: v.get("semantic_category") for item_id, v in meta.items()}


def main():
    types = load_item_types()
    valid_images = {p.stem for p in IMAGES_DIR.glob("*.jpg") if not p.stem.startswith("._")}

    with open(RAW_DIR / "nondisjoint" / "train.json") as f:
        outfits = json.load(f)

    def qualifies(item_id):
        return item_id in valid_images and types.get(item_id) is not None

    items_by_category = defaultdict(set)
    for outfit in outfits:
        for it in outfit["items"]:
            iid = it["item_id"]
            if qualifies(iid):
                items_by_category[types[iid]].add(iid)

    rng = random.Random(SEED)
    pools = {}
    for cat, items in items_by_category.items():
        items = sorted(items)
        pools[cat] = items if len(items) <= POOL_CAP else sorted(rng.sample(items, POOL_CAP))
    pool_membership = {cat: set(items) for cat, items in pools.items()}

    queries = []
    for outfit in outfits:
        item_ids = [it["item_id"] for it in outfit["items"]]
        qualifying = [i for i in item_ids if qualifies(i)]
        if len(qualifying) < 2:
            continue
        for target in qualifying:
            if target not in pool_membership[types[target]]:
                continue
            query_items = [i for i in qualifying if i != target]
            queries.append({
                "outfit_id": outfit["set_id"], "target_item": target,
                "category": types[target], "query_items": query_items,
            })

    print(f"Full train benchmark: {len(queries)} queries, "
          f"{sum(len(v) for v in pools.values())} pool slots.")

    sub_rng = random.Random(SEED)
    if len(queries) > SUBSAMPLE_QUERIES:
        queries_sub = sub_rng.sample(queries, SUBSAMPLE_QUERIES)
    else:
        queries_sub = queries

    with open(OUT_JSON, "w") as f:
        json.dump({"pools": pools, "queries": queries_sub}, f)
    print(f"Saved {OUT_JSON}: {len(queries_sub)} queries (subsampled from {len(queries)}), "
          f"{sum(len(v) for v in pools.values())} pool slots.")

    cat_counts_full = Counter(q["category"] for q in queries)
    cat_counts_sub = Counter(q["category"] for q in queries_sub)
    lines = [
        "# Phase 25: Train-Split CIR Benchmark Construction",
        "",
        "Built with the identical method as the validation/test benchmarks "
        "(candidate pools capped at 3,000/category, leave-one-out queries), "
        "pointed at `polyvore_raw/nondisjoint/train.json`. Used only for the "
        "per-epoch train-side Recall@10 diagnostic in phase 25's overfitting "
        "check (`overfitting_check.md`) -- never used as a training signal "
        "itself, and never compared against the actual test benchmark.",
        "",
        f"- Full construction: {len(queries)} leave-one-out queries, "
        f"{sum(len(v) for v in pools.values())} pool slots across {len(pools)} categories.",
        f"- Subsampled to {len(queries_sub)} queries (seed={SEED}) for per-epoch eval cost "
        "-- pools are kept at their full capped size, only the query set is subsampled.",
        "",
        "| Category | Pool size | Full queries | Subsampled queries |",
        "|---|---|---|---|",
    ]
    for cat in sorted(pools, key=lambda c: -len(pools[c])):
        lines.append(f"| {cat} | {len(pools[cat])} | {cat_counts_full.get(cat, 0)} | {cat_counts_sub.get(cat, 0)} |")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()

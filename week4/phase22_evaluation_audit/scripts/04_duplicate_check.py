"""
Phase 22, step 4: check for exact or near-exact duplicate images inside the
CIR benchmark's candidate pools, and specifically whether any query's own
target item has a duplicate-image twin sitting inside that same query's own
context items -- the most damaging form of leakage, since it would make that
specific query trivially solvable by any method that represents identical
photos as near-identical vectors, independent of genuine compatibility
understanding. Two independent checks: (1) exact byte-identical image files
(md5 hash of the actual .jpg), the strongest, cleanest signal; (2) near-exact
via raw frozen SigLIP embedding cosine similarity, which catches resized/
recompressed duplicates a byte hash would miss. Verifies directly rather than
assuming the phase 6-8 Amazon-data dedup work (a different dataset entirely)
carried over into this Polyvore-based benchmark.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

IMAGES_DIR = PHASE9_DIR / "data" / "images"
SIGLIP_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"

OUT_MD = BASE_DIR / "duplicate_check.md"
NEAR_DUP_THRESHOLD = 0.995


def load_benchmark():
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    pools, queries = load_benchmark()
    all_pool_items = set()
    for items in pools.values():
        all_pool_items.update(items)
    print(f"{len(all_pool_items)} unique items across all candidate pools.")

    # --- Check 1: exact byte-identical images among pool items ---
    hash_to_items = defaultdict(list)
    for item_id in sorted(all_pool_items):
        p = IMAGES_DIR / f"{item_id}.jpg"
        if p.exists():
            hash_to_items[md5_of(p)].append(item_id)
    exact_dup_groups = {h: items for h, items in hash_to_items.items() if len(items) > 1}
    n_exact_dup_items = sum(len(v) for v in exact_dup_groups.values())
    print(f"Exact byte-identical groups among pool items: {len(exact_dup_groups)} groups, "
          f"{n_exact_dup_items} items involved.")

    # --- Check 2: for every query, is the target item byte-identical to one
    # of the query's own context items? (the most damaging leakage form) ---
    item_to_hash = {}
    for h, items in hash_to_items.items():
        for it in items:
            item_to_hash[it] = h
    # also need hashes for query-context items that may not be in a pool
    all_query_context_items = set()
    for q in queries:
        all_query_context_items.update(q["query_items"])
    missing = all_query_context_items - set(item_to_hash.keys())
    for item_id in missing:
        p = IMAGES_DIR / f"{item_id}.jpg"
        if p.exists():
            item_to_hash[item_id] = md5_of(p)

    trivial_queries = []
    for qi, q in enumerate(queries):
        target = q["target_item"]
        if target not in item_to_hash:
            continue
        target_hash = item_to_hash[target]
        for ctx_item in q["query_items"]:
            if item_to_hash.get(ctx_item) == target_hash:
                trivial_queries.append((qi, q["outfit_id"], target, ctx_item))
                break
    print(f"Queries where target is byte-identical to one of its own context items: {len(trivial_queries)} / {len(queries)}")

    # --- Check 3: near-exact via raw SigLIP embedding cosine similarity,
    # within each category pool (catches resized/recompressed duplicates a
    # byte hash misses) ---
    data = np.load(SIGLIP_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    idx = {a: i for i, a in enumerate(item_ids)}

    near_dup_pair_counts = {}
    total_near_dup_pairs = 0
    for cat, items in pools.items():
        pool_idx = [idx[i] for i in items if i in idx]
        if len(pool_idx) < 2:
            continue
        emb = embeddings[pool_idx]
        sims = emb @ emb.T
        np.fill_diagonal(sims, 0.0)
        n_pairs = int((sims > NEAR_DUP_THRESHOLD).sum() // 2)
        near_dup_pair_counts[cat] = n_pairs
        total_near_dup_pairs += n_pairs
    print(f"Near-duplicate pairs (cosine > {NEAR_DUP_THRESHOLD}) within pools: {total_near_dup_pairs} total")

    # --- Check 4: near-exact between target and its own query context (same
    # embedding-based near-dup check, catches near-but-not-byte-identical
    # target/context leakage) ---
    near_dup_trivial_queries = 0
    for qi, q in enumerate(queries):
        target = q["target_item"]
        if target not in idx:
            continue
        t_emb = embeddings[idx[target]]
        for ctx_item in q["query_items"]:
            if ctx_item not in idx:
                continue
            sim = float(t_emb @ embeddings[idx[ctx_item]])
            if sim > NEAR_DUP_THRESHOLD:
                near_dup_trivial_queries += 1
                break

    lines = [
        "# Phase 22, Step 4: Duplicate / Near-Duplicate Images in the CIR Benchmark's Candidate Pools",
        "",
        f"Checked directly against the actual image files in "
        f"`week3/phase9_polyvore_compatibility/data/images/` and the raw frozen SigLIP "
        f"embeddings in `siglip_base.npz` -- not assumed carried over from the phases "
        f"6-8 Amazon-data dedup work, which was a different dataset (Amazon Clothing) "
        f"entirely and never touched Polyvore images.",
        "",
        "## Check 1: exact byte-identical images among pool items\n",
        f"{len(exact_dup_groups)} groups of byte-identical images found among the "
        f"{len(all_pool_items)} unique pool items, involving {n_exact_dup_items} items total "
        f"({100*n_exact_dup_items/len(all_pool_items):.2f}% of pool items).",
        "",
    ]
    if exact_dup_groups:
        lines.append("Largest groups (up to 10 shown):\n")
        lines.append("| Group size | Example item ids |")
        lines.append("|---|---|")
        for h, items in sorted(exact_dup_groups.items(), key=lambda kv: -len(kv[1]))[:10]:
            lines.append(f"| {len(items)} | {', '.join(items[:5])}{' ...' if len(items) > 5 else ''} |")
        lines.append("")

    lines.append("## Check 2: does any query's TARGET item byte-match one of its own CONTEXT items? (the critical leakage check)\n")
    lines.append(f"**{len(trivial_queries)} / {len(queries)} queries "
                  f"({100*len(trivial_queries)/len(queries):.3f}%)** have a target item that is "
                  f"byte-for-byte identical to one of the items already given as query context.")
    if trivial_queries:
        lines.append("")
        lines.append("Example affected queries (up to 10 shown):\n")
        lines.append("| Query index | Outfit id | Target item | Duplicate context item |")
        lines.append("|---|---|---|---|")
        for qi, outfit_id, target, ctx_item in trivial_queries[:10]:
            lines.append(f"| {qi} | {outfit_id} | {target} | {ctx_item} |")
    lines.append("")

    lines.append(f"## Check 3: near-exact duplicates within each category pool (raw SigLIP cosine > {NEAR_DUP_THRESHOLD})\n")
    lines.append("| Category | Pool size | Near-duplicate pairs |")
    lines.append("|---|---|---|")
    for cat in sorted(pools, key=lambda c: -len(pools[c])):
        lines.append(f"| {cat} | {len(pools[cat])} | {near_dup_pair_counts.get(cat, 0)} |")
    lines.append("")
    lines.append(f"**{total_near_dup_pairs} total near-duplicate pairs** across all pools "
                  f"(includes the exact byte-identical pairs from Check 1, since those trivially "
                  f"also pass a >{NEAR_DUP_THRESHOLD} cosine threshold).")
    lines.append("")

    lines.append("## Check 4: near-exact (not just byte-exact) target/context leakage\n")
    lines.append(f"**{near_dup_trivial_queries} / {len(queries)} queries "
                  f"({100*near_dup_trivial_queries/len(queries):.3f}%)** have a target item with "
                  f"raw-SigLIP cosine similarity > {NEAR_DUP_THRESHOLD} to one of its own context "
                  f"items -- this superset includes Check 2's byte-identical cases plus any "
                  f"resized/recompressed near-duplicates a byte hash would miss.")
    lines.append("")

    lines.append("## Verdict\n")
    rate = len(trivial_queries) / len(queries)
    if rate < 0.005 and total_near_dup_pairs / max(len(all_pool_items), 1) < 0.01:
        lines.append(
            f"**Duplicate/near-duplicate contamination is real but negligible at benchmark "
            f"scale.** {len(trivial_queries)} queries ({100*rate:.3f}%) are trivially solvable "
            f"via target/context duplication, and near-duplicate pairs are a small fraction "
            f"of total pool items. At this rate, duplicate leakage could not plausibly explain "
            f"phase 9's roughly 2x advantage over the next-best 'mine' configuration (phase "
            f"13b's CSA-Net reproduction) -- removing {100*rate:.3f}% of queries would not move "
            f"Recall@10 by an amount close to the gap being explained. This is reported as a "
            f"known minor imperfection in the benchmark construction, not a fatal flaw."
        )
    else:
        lines.append(
            f"**Duplicate contamination is high enough to be a real concern** "
            f"({100*rate:.3f}% of queries trivially solvable via target/context duplication). "
            f"This should be treated as a genuine caveat on the current comparison, not rounded "
            f"away -- see phase22_notes.md for how this affects the overall verdict."
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

"""
Phase 11, steps 2-4: build every blended/diversity-forcing configuration and
evaluate all of them three ways (retrieval accuracy, category diversity,
popularity behavior) on the same untouched 1,872-product Amazon eval sample.

Step 2 (blending): blended_sim = alpha * raw_visual_sim + (1 - alpha) *
compatibility_sim, computed separately for phase 8's and phase 9's
compatibility signals (not mixed together), swept at alpha in {0.3, 0.5, 0.7}.

Step 3 (diversity-forcing): built independently of blending. Per query: the
top-4 items by raw SigLIP similarity, plus one slot reserved for the
highest-raw-similarity item of a DIFFERENT fine-grained type (categories
breadcrumb index 3, same level and same documented caveats as phase 8) than
the query. This defines the forced top-5. For top-10 evaluation, this same
single diversity slot is kept at rank 5, and ranks 6-10 continue filling in
from the raw SigLIP ranking (excluding whatever is already used) -- the brief
only specifies a forced slot in the top-5, so this is the most literal
extension to K=10 (one diversity slot total, not a second one manufactured
for the second half of the list). Two edge cases are tracked and reported
explicitly rather than silently patched: queries whose own type is unknown
(can't tell what counts as "different"), and queries with zero cross-type
candidates anywhere in the sample (would require silently falling back to a
same-type item, which the brief says not to do) -- both fall back to the
plain raw-SigLIP top-10 for that one query, and are counted.

Step 4 (evaluation): every configuration (raw SigLIP alone, phase 8 alone,
phase 9 alone, the six blends, the diversity-forcing configuration) is run
through the same three metric families:
  1. Hit Rate@5/10, Precision@5/10 against also_buy, full view AND the
     cross-type-only view used since phase 8 (identical ground truth
     construction, so directly comparable to every prior phase's numbers).
  2. Category diversity: mean fraction of top-5/top-10 that are a different
     fine-grained type than the query, averaged over queries with a known
     own type (unknown-type queries are excluded from this average, same
     reasoning as the cross-type ground truth view).
  3. Popularity: ARP@5/10 and catalog coverage@5/10, reusing phase 3's exact
     catalog-wide popularity_lookup.csv.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"
PHASE3_LOOKUP_CSV = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"
BUNDLE_NPZ = BASE_DIR / "data" / "embeddings_bundle.npz"

RESULTS_MD = BASE_DIR / "results_table.md"
DIVERSITY_STATS_JSON = BASE_DIR / "data" / "diversity_forcing_stats.json"

TYPE_INDEX = 3
K_VALUES = [5, 10]
ALPHAS = [0.3, 0.5, 0.7]


def load_eval_data():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    also_buy, types = {}, {}
    for _, row in df.iterrows():
        also_buy[row["asin"]] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()
        cats = ast.literal_eval(row["category"])
        types[row["asin"]] = cats[TYPE_INDEX] if len(cats) > TYPE_INDEX else None
    return also_buy, types


def build_cross_type_ground_truth(asins, also_buy, types):
    cross_type_gt = {}
    n_query_unknown = 0
    for a in asins:
        my_type = types.get(a)
        if my_type is None:
            n_query_unknown += 1
            continue
        related = also_buy.get(a, set())
        cross_type_gt[a] = {r for r in related if types.get(r) is not None and types.get(r) != my_type}
    return cross_type_gt, n_query_unknown


def topk_from_sims(sims, asins, k_total=10):
    s = sims.copy()
    np.fill_diagonal(s, -np.inf)
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    ranked_lists = {}
    for q in asins:
        row = s[asin_to_idx[q]]
        top_idx = np.argsort(-row)[:k_total]
        ranked_lists[q] = [asins[j] for j in top_idx]
    return ranked_lists


def build_diversity_forced_lists(asins, raw_sims, types, k_total=10):
    s = raw_sims.copy()
    np.fill_diagonal(s, -np.inf)
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    ranked_lists = {}
    stats = {"ok": 0, "unknown_own_type": 0, "no_cross_type_candidate": 0}

    for q in asins:
        row = s[asin_to_idx[q]]
        order = np.argsort(-row)
        raw_ranked = [asins[j] for j in order if asins[j] != q]

        my_type = types.get(q)
        if my_type is None:
            ranked_lists[q] = raw_ranked[:k_total]
            stats["unknown_own_type"] += 1
            continue

        top4 = raw_ranked[:4]
        top4_set = set(top4)
        diverse_item = None
        for cand in raw_ranked:
            if cand in top4_set:
                continue
            if types.get(cand) is not None and types.get(cand) != my_type:
                diverse_item = cand
                break

        if diverse_item is None:
            ranked_lists[q] = raw_ranked[:k_total]
            stats["no_cross_type_candidate"] += 1
            continue

        used = top4_set | {diverse_item}
        remaining = [a for a in raw_ranked if a not in used]
        ranked_lists[q] = top4 + [diverse_item] + remaining[: k_total - 5]
        stats["ok"] += 1

    return ranked_lists, stats


def evaluate_accuracy(ranked_lists, ground_truth, query_asins):
    n = len(query_asins)
    hits_at = {k: 0 for k in K_VALUES}
    precision_sum = {k: 0.0 for k in K_VALUES}
    for q in query_asins:
        related = ground_truth.get(q, set())
        retrieved_all = ranked_lists[q]
        for k in K_VALUES:
            retrieved_k = retrieved_all[:k]
            n_hits = sum(1 for r in retrieved_k if r in related)
            if n_hits > 0:
                hits_at[k] += 1
            precision_sum[k] += n_hits / k
    return {
        **{f"hit_rate@{k}": hits_at[k] / n for k in K_VALUES},
        **{f"precision@{k}": precision_sum[k] / n for k in K_VALUES},
    }


def evaluate_diversity(ranked_lists, types, asins):
    used_queries = [q for q in asins if types.get(q) is not None]
    n = len(used_queries)
    frac_sum = {k: 0.0 for k in K_VALUES}
    for q in used_queries:
        my_type = types[q]
        retrieved_all = ranked_lists[q]
        for k in K_VALUES:
            retrieved_k = retrieved_all[:k]
            n_cross = sum(1 for r in retrieved_k if types.get(r) is not None and types.get(r) != my_type)
            frac_sum[k] += n_cross / k
    return {**{f"diversity@{k}": frac_sum[k] / n for k in K_VALUES}, "n_queries_used": n}


def evaluate_popularity(ranked_lists, ref_count, asins):
    top5_pool, top10_pool = [], []
    for q in asins:
        retrieved_all = ranked_lists[q]
        top5_pool.extend(retrieved_all[:5])
        top10_pool.extend(retrieved_all[:10])
    return {
        "arp@5": float(np.mean([ref_count.get(a, 0) for a in top5_pool])),
        "arp@10": float(np.mean([ref_count.get(a, 0) for a in top10_pool])),
        "coverage@5": len(set(top5_pool)) / len(asins),
        "coverage@10": len(set(top10_pool)) / len(asins),
    }


def main():
    also_buy, types = load_eval_data()
    bundle = np.load(BUNDLE_NPZ, allow_pickle=True)
    asins = [str(a) for a in bundle["asins"]]
    raw, p8, p9 = bundle["raw"], bundle["phase8"], bundle["phase9"]
    print(f"Loaded bundle: {len(asins)} products, raw/phase8/phase9 embeddings.")

    n_known_type = sum(1 for a in asins if types.get(a) is not None)
    print(f"Eval products with known type at index {TYPE_INDEX}: {n_known_type}/{len(asins)}")

    cross_type_gt, n_query_unknown = build_cross_type_ground_truth(asins, also_buy, types)
    cross_type_queries = [a for a in asins if types.get(a) is not None]
    n_cross_edges = sum(len(v) for v in cross_type_gt.values())
    print(f"Cross-type-only ground truth: {n_cross_edges} edges across {len(cross_type_queries)} queries "
          f"({n_query_unknown} excluded, unknown own type) -- identical to phase 8/9/10's view.")

    ref_df = pd.read_csv(PHASE3_LOOKUP_CSV)
    ref_count = dict(zip(ref_df["asin"], ref_df["ref_count"]))

    raw_sims = raw @ raw.T
    p8_sims = p8 @ p8.T
    p9_sims = p9 @ p9.T

    sim_configs = [("Raw SigLIP (alone)", raw_sims), ("Phase 8 alone (Amazon-trained)", p8_sims),
                   ("Phase 9 alone (Polyvore-trained)", p9_sims)]
    for alpha in ALPHAS:
        sim_configs.append((f"Blend: alpha={alpha} raw + Phase 8 ({alpha:.1f} raw / {1-alpha:.1f} compat)",
                             alpha * raw_sims + (1 - alpha) * p8_sims))
    for alpha in ALPHAS:
        sim_configs.append((f"Blend: alpha={alpha} raw + Phase 9 ({alpha:.1f} raw / {1-alpha:.1f} compat)",
                             alpha * raw_sims + (1 - alpha) * p9_sims))

    all_results = []
    for label, sims in sim_configs:
        ranked_lists = topk_from_sims(sims, asins)
        full_acc = evaluate_accuracy(ranked_lists, also_buy, asins)
        cross_acc = evaluate_accuracy(ranked_lists, cross_type_gt, cross_type_queries)
        div = evaluate_diversity(ranked_lists, types, asins)
        pop = evaluate_popularity(ranked_lists, ref_count, asins)
        all_results.append((label, full_acc, cross_acc, div, pop))
        print(f"{label}: full HR@5={full_acc['hit_rate@5']:.3f} cross HR@5={cross_acc['hit_rate@5']:.3f} "
              f"div@5={div['diversity@5']:.3f} ARP@5={pop['arp@5']:.1f} cov@5={pop['coverage@5']:.3f}")

    div_ranked_lists, div_stats = build_diversity_forced_lists(asins, raw_sims, types)
    print(f"Diversity-forcing construction stats: {div_stats}")
    full_acc = evaluate_accuracy(div_ranked_lists, also_buy, asins)
    cross_acc = evaluate_accuracy(div_ranked_lists, cross_type_gt, cross_type_queries)
    div = evaluate_diversity(div_ranked_lists, types, asins)
    pop = evaluate_popularity(div_ranked_lists, ref_count, asins)
    all_results.append(("Diversity-forcing (top-4 raw + 1 forced cross-type slot)", full_acc, cross_acc, div, pop))
    print(f"Diversity-forcing: full HR@5={full_acc['hit_rate@5']:.3f} cross HR@5={cross_acc['hit_rate@5']:.3f} "
          f"div@5={div['diversity@5']:.3f} ARP@5={pop['arp@5']:.1f} cov@5={pop['coverage@5']:.3f}")

    DIVERSITY_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DIVERSITY_STATS_JSON.write_text(json.dumps(div_stats, indent=2))

    lines = [
        "# Phase 11: Blended Similarity and Diversity-Forcing Retrieval -- Results",
        "",
        f"Eval sample: same untouched 1,872-product Amazon sample used since phase 1b. "
        f"Cross-type-only ground truth: {n_cross_edges} edges across {len(cross_type_queries)} queries "
        f"({n_query_unknown} excluded, unknown own type at categories-breadcrumb index {TYPE_INDEX} -- "
        f"identical to phase 8/9/10's view). Category diversity metric excludes the same "
        f"{n_query_unknown} unknown-own-type queries from its average, for the same reason.",
        "",
        f"Diversity-forcing construction: {div_stats['ok']} queries got a genuine forced cross-type "
        f"slot; {div_stats['unknown_own_type']} fell back to plain raw-SigLIP ranking because the "
        f"query's own type is unknown; {div_stats['no_cross_type_candidate']} fell back because no "
        f"cross-type candidate existed anywhere in the sample for that query (neither case was "
        f"silently filled with a same-type item).",
        "",
        "Blend formula: `blended = alpha * raw_visual_similarity + (1 - alpha) * compatibility_similarity`. "
        "alpha=0.7 means mostly raw visual similarity; alpha=0.3 means mostly the compatibility signal.",
        "",
        "## 1. Retrieval accuracy -- full also_buy ground truth (N=1,872 queries)",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, full_acc, _, _, _ in all_results:
        lines.append(f"| {label} | {full_acc['hit_rate@5']:.3f} | {full_acc['hit_rate@10']:.3f} "
                      f"| {full_acc['precision@5']:.3f} | {full_acc['precision@10']:.3f} |")

    lines += [
        "",
        f"## 2. Retrieval accuracy -- cross-type-only ground truth (N={len(cross_type_queries)} queries, "
        f"{n_cross_edges} edges)",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, _, cross_acc, _, _ in all_results:
        lines.append(f"| {label} | {cross_acc['hit_rate@5']:.3f} | {cross_acc['hit_rate@10']:.3f} "
                      f"| {cross_acc['precision@5']:.3f} | {cross_acc['precision@10']:.3f} |")

    lines += [
        "",
        "## 3. Category diversity -- mean fraction of top-K that is a different fine-grained type "
        f"than the query (N={all_results[0][3]['n_queries_used']} queries with known own type)",
        "",
        "| Configuration | Diversity@5 | Diversity@10 |",
        "|---|---|---|",
    ]
    for label, _, _, div, _ in all_results:
        lines.append(f"| {label} | {div['diversity@5']:.3f} | {div['diversity@10']:.3f} |")

    lines += [
        "",
        "## 4. Popularity behavior (ARP = mean catalog-wide reference count; coverage = fraction of the "
        "1,872-product sample recommended at least once)",
        "",
        "| Configuration | ARP@5 | ARP@10 | Coverage@5 | Coverage@10 |",
        "|---|---|---|---|---|",
    ]
    for label, _, _, _, pop in all_results:
        lines.append(f"| {label} | {pop['arp@5']:.2f} | {pop['arp@10']:.2f} "
                      f"| {pop['coverage@5']:.3f} | {pop['coverage@10']:.3f} |")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")
    print(f"Saved {DIVERSITY_STATS_JSON}")


if __name__ == "__main__":
    main()

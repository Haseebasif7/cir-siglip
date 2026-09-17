import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_MD = BASE_DIR / "results_table.md"
CATEGORY_BREAKDOWN_MD = BASE_DIR / "category_breakdown.md"
REPORT_SUMMARY_MD = BASE_DIR / "report_summary_table.md"
RETRIEVAL_JSON = BASE_DIR / "data" / "retrieval_results.json"

K_VALUES = [5, 10]
TECHNIQUES = ["resnet50", "clip_vit_b32", "fashionclip", "siglip_base"]
TECHNIQUE_LABELS = {
    "resnet50": "ResNet50",
    "clip_vit_b32": "CLIP ViT-B/32",
    "fashionclip": "FashionCLIP",
    "siglip_base": "SigLIP",
}

def load_relatedness():
    df = pd.read_csv(SAMPLE_CSV)
    related = {}
    category = {}
    for _, row in df.iterrows():
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        also_viewed = ast.literal_eval(row["also_viewed"]) if pd.notna(row["also_viewed"]) else []
        related[row["asin"]] = set(also_buy) | set(also_viewed)
        category[row["asin"]] = row["category_bucket"]
    return related, category

def compute_category_overlap(related, category):
    asins = set(category.keys())
    per_cat = defaultdict(lambda: {"overlap": 0, "total": 0})
    for asin, related_set in related.items():
        cat = category.get(asin, "(unknown)")
        overlap = len(related_set & asins)
        per_cat[cat]["overlap"] += overlap
        per_cat[cat]["total"] += len(related_set)
    return per_cat

def evaluate_technique(technique, related, category):
    data = np.load(EMBEDDINGS_DIR / f"{technique}.npz", allow_pickle=True)
    asins = data["asins"]
    embeddings = data["embeddings"]

    sims = embeddings @ embeddings.T
    n = len(asins)
    np.fill_diagonal(sims, -np.inf)

    max_k = max(K_VALUES)
    top_idx = np.argsort(-sims, axis=1)[:, :max_k]

    per_query = []
    hits_at = {k: 0 for k in K_VALUES}
    precision_sum = {k: 0.0 for k in K_VALUES}
    cat_hits = defaultdict(lambda: {k: 0 for k in K_VALUES})
    cat_precision_sum = defaultdict(lambda: {k: 0.0 for k in K_VALUES})
    cat_n = defaultdict(int)

    for i in range(n):
        query_asin = asins[i]
        query_cat = category.get(query_asin, "(unknown)")
        related_set = related.get(query_asin, set())
        retrieved_all = [asins[j] for j in top_idx[i]]
        cat_n[query_cat] += 1

        query_hits = {}
        for k in K_VALUES:
            retrieved_k = retrieved_all[:k]
            hit_flags = [1 if r in related_set else 0 for r in retrieved_k]
            n_hits = sum(hit_flags)
            if n_hits > 0:
                hits_at[k] += 1
                cat_hits[query_cat][k] += 1
            precision_sum[k] += n_hits / k
            cat_precision_sum[query_cat][k] += n_hits / k
            query_hits[k] = hit_flags

        per_query.append({
            "asin": str(query_asin),
            "category": query_cat,
            "retrieved": [str(a) for a in retrieved_all],
            "hit_flags_at_5": query_hits[5],
            "hit_flags_at_10": query_hits[10],
            "n_hits_at_5": sum(query_hits[5]),
        })

    metrics = {}
    for k in K_VALUES:
        metrics[f"hit_rate@{k}"] = hits_at[k] / n
        metrics[f"precision@{k}"] = precision_sum[k] / n

    category_metrics = {}
    for cat, n_q in cat_n.items():
        category_metrics[cat] = {
            "n_queries": n_q,
            **{f"hit_rate@{k}": cat_hits[cat][k] / n_q for k in K_VALUES},
            **{f"precision@{k}": cat_precision_sum[cat][k] / n_q for k in K_VALUES},
        }

    return metrics, category_metrics, per_query

def write_report_summary(all_category_metrics, category_overlap, category_order):
    lines = [
        "# Phase 1b: Report-Ready Summary Table",
        "",
        "One row per category: sample size, ground-truth density (overlap%),",
        "and Hit Rate@5 for all four techniques side by side, plus which",
        "technique won. Meant to be dropped directly into the technical report",
        "-- results_table.md and category_breakdown.md have the same numbers",
        "split across files; this puts them in one place.",
        "",
    ]
    header = "| Category | N queries | Overlap refs | Overlap % | " + \
             " | ".join(f"{TECHNIQUE_LABELS[t]} HR@5" for t in TECHNIQUES) + \
             " | Best technique |"
    sep = "|---|---|---|---|" + "---|" * len(TECHNIQUES) + "---|"
    lines.append(header)
    lines.append(sep)

    for cat in category_order:
        ov = category_overlap.get(cat, {"overlap": 0, "total": 0})
        pct = 100 * ov["overlap"] / ov["total"] if ov["total"] else 0.0
        n_queries = None
        hr5_values = {}
        for t in TECHNIQUES:
            cm = all_category_metrics.get(t, {}).get(cat)
            if cm is None:
                continue
            n_queries = cm["n_queries"]
            hr5_values[t] = cm["hit_rate@5"]
        if n_queries is None:
            continue
        best = max(hr5_values, key=hr5_values.get) if hr5_values else None
        row = f"| {cat} | {n_queries} | {ov['overlap']} | {pct:.1f}% | "
        row += " | ".join(f"{hr5_values.get(t, float('nan')):.3f}" for t in TECHNIQUES)
        row += f" | {TECHNIQUE_LABELS.get(best, '-')} |"
        lines.append(row)

    REPORT_SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_SUMMARY_MD}")

def main():
    related, category = load_relatedness()
    category_overlap = compute_category_overlap(related, category)

    all_metrics = {}
    all_category_metrics = {}
    all_retrieval = {}

    for technique in TECHNIQUES:
        npz_path = EMBEDDINGS_DIR / f"{technique}.npz"
        if not npz_path.exists():
            print(f"Skipping {technique}: {npz_path} not found (run step 02 first)")
            continue
        metrics, category_metrics, per_query = evaluate_technique(technique, related, category)
        all_metrics[technique] = metrics
        all_category_metrics[technique] = category_metrics
        all_retrieval[technique] = per_query
        print(f"{technique}: {metrics}")

    with open(RETRIEVAL_JSON, "w") as f:
        json.dump(all_retrieval, f)

    lines = [
        "# Phase 1b Results: Frozen Embedding Retrieval Comparison (Category-Balanced Sample, quota=175)",
        "",
        "| Technique | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for technique, m in all_metrics.items():
        lines.append(
            f"| {technique} | {m['hit_rate@5']:.3f} | {m['hit_rate@10']:.3f} "
            f"| {m['precision@5']:.3f} | {m['precision@10']:.3f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")

    category_order = sorted(
        {cat for cm in all_category_metrics.values() for cat in cm},
        key=lambda c: -max(cm.get(c, {}).get("n_queries", 0) for cm in all_category_metrics.values())
    )
    cat_lines = ["# Phase 1b: Hit Rate by Category, per Technique", ""]
    for technique in TECHNIQUES:
        if technique not in all_category_metrics:
            continue
        cat_lines.append(f"## {technique}")
        cat_lines.append("")
        cat_lines.append("| Category | N queries | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |")
        cat_lines.append("|---|---|---|---|---|---|")
        cm = all_category_metrics[technique]
        for cat in category_order:
            if cat not in cm:
                continue
            d = cm[cat]
            cat_lines.append(
                f"| {cat} | {d['n_queries']} | {d['hit_rate@5']:.3f} | {d['hit_rate@10']:.3f} "
                f"| {d['precision@5']:.3f} | {d['precision@10']:.3f} |"
            )
        cat_lines.append("")
    CATEGORY_BREAKDOWN_MD.write_text("\n".join(cat_lines) + "\n")
    print(f"Saved {CATEGORY_BREAKDOWN_MD}")

    write_report_summary(all_category_metrics, category_overlap, category_order)

if __name__ == "__main__":
    main()

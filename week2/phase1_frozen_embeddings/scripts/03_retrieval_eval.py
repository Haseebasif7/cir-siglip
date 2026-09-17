import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_MD = BASE_DIR / "results_table.md"
RETRIEVAL_JSON = BASE_DIR / "data" / "retrieval_results.json"

K_VALUES = [5, 10]
TECHNIQUES = ["resnet50", "clip_vit_b32", "fashionclip", "siglip_base"]

def load_relatedness():
    df = pd.read_csv(SAMPLE_CSV)
    related = {}
    for _, row in df.iterrows():
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        also_viewed = ast.literal_eval(row["also_viewed"]) if pd.notna(row["also_viewed"]) else []
        related[row["asin"]] = set(also_buy) | set(also_viewed)
    return related

def evaluate_technique(technique, related):
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

    for i in range(n):
        query_asin = asins[i]
        related_set = related.get(query_asin, set())
        retrieved_all = [asins[j] for j in top_idx[i]]

        query_hits = {}
        for k in K_VALUES:
            retrieved_k = retrieved_all[:k]
            hit_flags = [1 if r in related_set else 0 for r in retrieved_k]
            n_hits = sum(hit_flags)
            if n_hits > 0:
                hits_at[k] += 1
            precision_sum[k] += n_hits / k
            query_hits[k] = hit_flags

        per_query.append({
            "asin": str(query_asin),
            "retrieved": [str(a) for a in retrieved_all],
            "hit_flags_at_5": query_hits[5],
            "hit_flags_at_10": query_hits[10],
            "n_hits_at_5": sum(query_hits[5]),
        })

    metrics = {}
    for k in K_VALUES:
        metrics[f"hit_rate@{k}"] = hits_at[k] / n
        metrics[f"precision@{k}"] = precision_sum[k] / n

    return metrics, per_query

def main():
    related = load_relatedness()
    all_metrics = {}
    all_retrieval = {}

    for technique in TECHNIQUES:
        npz_path = EMBEDDINGS_DIR / f"{technique}.npz"
        if not npz_path.exists():
            print(f"Skipping {technique}: {npz_path} not found (run step 02 first)")
            continue
        metrics, per_query = evaluate_technique(technique, related)
        all_metrics[technique] = metrics
        all_retrieval[technique] = per_query
        print(f"{technique}: {metrics}")

    with open(RETRIEVAL_JSON, "w") as f:
        json.dump(all_retrieval, f)

    lines = [
        "# Phase 1 Results: Frozen Embedding Retrieval Comparison",
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

if __name__ == "__main__":
    main()

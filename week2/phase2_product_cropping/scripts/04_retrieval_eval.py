import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1B_DIR = BASE_DIR.parent / "phase1b_category_balanced"
SAMPLE_CSV = PHASE1B_DIR / "data" / "sample_data.csv"
PHASE2_EMB_DIR = BASE_DIR / "embeddings"
PHASE1B_EMB_DIR = PHASE1B_DIR / "embeddings"
CROPS_A_DIR = BASE_DIR / "data" / "crops_a"
CROPS_B_DIR = BASE_DIR / "data" / "crops_b"

RESULTS_MD = BASE_DIR / "results_table.md"
CATEGORY_BREAKDOWN_MD = BASE_DIR / "category_breakdown.md"
RETRIEVAL_JSON = BASE_DIR / "data" / "retrieval_results.json"
COMMON_SET_LOG = BASE_DIR / "data" / "common_evaluable_set.md"

K_VALUES = [5, 10]
ENCODERS = ["siglip_base", "fashionclip"]
ENCODER_LABELS = {"siglip_base": "SigLIP", "fashionclip": "FashionCLIP"}
VARIANTS = ["baseline", "cropA", "cropB"]
VARIANT_LABELS = {"baseline": "Uncropped (baseline)", "cropA": "Method A (rembg)", "cropB": "Method B (Grounding DINO)"}

def load_relatedness_and_category():
    df = pd.read_csv(SAMPLE_CSV)
    related = {}
    category = {}
    for _, row in df.iterrows():
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        also_viewed = ast.literal_eval(row["also_viewed"]) if pd.notna(row["also_viewed"]) else []
        asin = str(row["asin"])
        related[asin] = set(also_buy) | set(also_viewed)
        category[asin] = row["category_bucket"]
    return related, category

def compute_common_evaluable_set():
    crop_a_asins = {p.stem for p in CROPS_A_DIR.glob("*.jpg")}
    crop_b_asins = {p.stem for p in CROPS_B_DIR.glob("*.jpg")}
    common = crop_a_asins & crop_b_asins
    return common, crop_a_asins, crop_b_asins

def load_embeddings_dict(path):
    data = np.load(path, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    return dict(zip(asins, embeddings))

def get_embedding_path(encoder, variant):
    if variant == "baseline":
        return PHASE1B_EMB_DIR / f"{encoder}.npz"
    elif variant == "cropA":
        return PHASE2_EMB_DIR / f"{encoder}_cropA.npz"
    elif variant == "cropB":
        return PHASE2_EMB_DIR / f"{encoder}_cropB.npz"
    raise ValueError(variant)

def evaluate(asins_ordered, embeddings, related, category):
    n = len(asins_ordered)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)

    max_k = max(K_VALUES)
    top_idx = np.argsort(-sims, axis=1)[:, :max_k]

    hits_at = {k: 0 for k in K_VALUES}
    precision_sum = {k: 0.0 for k in K_VALUES}
    cat_hits = defaultdict(lambda: {k: 0 for k in K_VALUES})
    cat_precision_sum = defaultdict(lambda: {k: 0.0 for k in K_VALUES})
    cat_n = defaultdict(int)
    per_query = []

    for i in range(n):
        query_asin = asins_ordered[i]
        query_cat = category.get(query_asin, "(unknown)")
        related_set = related.get(query_asin, set())
        retrieved_all = [asins_ordered[j] for j in top_idx[i]]
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
            "asin": query_asin,
            "category": query_cat,
            "retrieved": retrieved_all,
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

def main():
    related, category = load_relatedness_and_category()
    common_set, crop_a_asins, crop_b_asins = compute_common_evaluable_set()
    common_ordered = sorted(common_set)
    n_common = len(common_ordered)

    COMMON_SET_LOG.write_text(
        "# Phase 2: Common Evaluable Set\n\n"
        f"Full phase 1b sample: 1872 products.\n"
        f"Method A (rembg) successful crops: {len(crop_a_asins)}\n"
        f"Method B (Grounding DINO) successful crops: {len(crop_b_asins)}\n"
        f"Common evaluable set (A ∩ B, used for ALL variants incl. baseline "
        f"so candidate-pool size is identical everywhere): {n_common} "
        f"({100*n_common/1872:.1f}% of the full sample)\n"
    )
    print(f"Common evaluable set: {n_common} / 1872 products")

    all_metrics = {}
    all_category_metrics = {}
    all_retrieval = {}
    phase1b_reference_metrics = {}

    for encoder in ENCODERS:
        full_dict = load_embeddings_dict(PHASE1B_EMB_DIR / f"{encoder}.npz")
        full_asins = sorted(full_dict.keys())
        full_emb = np.stack([full_dict[a] for a in full_asins])
        ref_metrics, _, _ = evaluate(full_asins, full_emb, related, category)
        phase1b_reference_metrics[encoder] = ref_metrics

        for variant in VARIANTS:
            key = f"{encoder}_{variant}"
            emb_dict = load_embeddings_dict(get_embedding_path(encoder, variant))
            missing = [a for a in common_ordered if a not in emb_dict]
            if missing:
                raise RuntimeError(f"{key}: {len(missing)} common-set asins missing embeddings, e.g. {missing[:5]}")
            embeddings = np.stack([emb_dict[a] for a in common_ordered])

            metrics, category_metrics, per_query = evaluate(common_ordered, embeddings, related, category)
            all_metrics[key] = metrics
            all_category_metrics[key] = category_metrics
            all_retrieval[key] = per_query
            print(f"{key}: {metrics}")

    with open(RETRIEVAL_JSON, "w") as f:
        json.dump(all_retrieval, f)

    lines = [
        "# Phase 2 Results: Product-Region Cropping vs Uncropped Baseline",
        "",
        f"All numbers below (baseline, Method A, Method B) are computed on "
        f"the same {n_common}-product common evaluable set (see "
        f"data/common_evaluable_set.md) so Hit Rate@K/Precision@K are "
        f"directly comparable across variants -- see script docstring for why.",
        "",
        "| Encoder | Variant | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|---|",
    ]
    for encoder in ENCODERS:
        for variant in VARIANTS:
            m = all_metrics[f"{encoder}_{variant}"]
            lines.append(
                f"| {ENCODER_LABELS[encoder]} | {VARIANT_LABELS[variant]} | "
                f"{m['hit_rate@5']:.3f} | {m['hit_rate@10']:.3f} | "
                f"{m['precision@5']:.3f} | {m['precision@10']:.3f} |"
            )
    lines.append("")
    lines.append(f"## For reference: original phase 1b baseline (full 1,872-product pool, not the {n_common}-product common set)")
    lines.append("")
    lines.append("| Encoder | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |")
    lines.append("|---|---|---|---|---|")
    for encoder in ENCODERS:
        m = phase1b_reference_metrics[encoder]
        lines.append(
            f"| {ENCODER_LABELS[encoder]} | {m['hit_rate@5']:.3f} | {m['hit_rate@10']:.3f} "
            f"| {m['precision@5']:.3f} | {m['precision@10']:.3f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")

    category_order = sorted(
        {cat for cm in all_category_metrics.values() for cat in cm},
        key=lambda c: -max(cm.get(c, {}).get("n_queries", 0) for cm in all_category_metrics.values())
    )
    cat_lines = ["# Phase 2: Hit Rate by Category, per Encoder x Variant", ""]
    for encoder in ENCODERS:
        cat_lines.append(f"## {ENCODER_LABELS[encoder]}")
        cat_lines.append("")
        header = "| Category | N queries | " + " | ".join(
            f"{VARIANT_LABELS[v]} HR@5" for v in VARIANTS
        ) + " | " + " | ".join(
            f"{VARIANT_LABELS[v]} HR@10" for v in VARIANTS
        ) + " |"
        cat_lines.append(header)
        cat_lines.append("|---|---|" + "---|" * len(VARIANTS) + "---|" * len(VARIANTS))
        for cat in category_order:
            cms = {v: all_category_metrics[f"{encoder}_{v}"].get(cat) for v in VARIANTS}
            if any(c is None for c in cms.values()):
                continue
            n_q = cms["baseline"]["n_queries"]
            row = f"| {cat} | {n_q} | "
            row += " | ".join(f"{cms[v]['hit_rate@5']:.3f}" for v in VARIANTS)
            row += " | "
            row += " | ".join(f"{cms[v]['hit_rate@10']:.3f}" for v in VARIANTS)
            row += " |"
            cat_lines.append(row)
        cat_lines.append("")
    CATEGORY_BREAKDOWN_MD.write_text("\n".join(cat_lines) + "\n")
    print(f"Saved {CATEGORY_BREAKDOWN_MD}")

if __name__ == "__main__":
    main()

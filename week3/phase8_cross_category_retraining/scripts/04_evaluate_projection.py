"""
Phase 8, step 4: evaluate two ways on the exact same untouched 1,872-product
eval sample used by every prior phase.

1. Overall Hit Rate@5/10, Precision@5/10 against the FULL also_buy ground
   truth -- identical methodology to phase 7's step 6, so directly
   comparable to raw SigLIP and phase 7's Model A/B numbers in one table.
2. A second, targeted score computed only against the subset of ground-truth
   edges that are themselves heterogeneous dyads (cross-type, same index-3
   type level from step 1) -- isolates whether the fix works for the
   specific thing this model was trained to predict, even if the aggregate
   number doesn't move.

Recomputes ALL five configurations (raw SigLIP, phase 7 Model A/B, phase 8
Model A/B) against BOTH ground-truth views for a fair, consistent
comparison -- the cross-type-only view is new and wasn't computed for phase
7's models before, so it needs to be run for all five here, not just phase
8's two.

Query products whose own type is unknown (breadcrumb too shallow, see step
1) are excluded from the cross-type-only view's denominator entirely (N is
reported explicitly) -- we cannot determine what counts as "different type"
relative to a query whose own type isn't known. Ground-truth targets with
unknown type are conservatively NOT counted as cross-type for the same
reason step 2 used (can't confirm a type difference without knowing both
sides), even when the query's own type is known.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent / "phase7_learned_compatibility"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"

RESULTS_MD = BASE_DIR / "results_table.md"

TYPE_INDEX = 3  # same level chosen in step 1
EXPECTED_BASELINE = {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144}
BASELINE_TOLERANCE = 0.001

K_VALUES = [5, 10]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_eval_data():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    also_buy = {}
    types = {}
    for _, row in df.iterrows():
        also_buy[row["asin"]] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()
        cats = ast.literal_eval(row["category"])
        types[row["asin"]] = cats[TYPE_INDEX] if len(cats) > TYPE_INDEX else None

    data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return asins, embeddings, also_buy, types


def build_cross_type_ground_truth(asins, also_buy, types):
    cross_type_gt = {}
    n_query_unknown = 0
    for a in asins:
        my_type = types.get(a)
        if my_type is None:
            n_query_unknown += 1
            continue
        related = also_buy.get(a, set())
        cross = {r for r in related if types.get(r) is not None and types.get(r) != my_type}
        cross_type_gt[a] = cross
    return cross_type_gt, n_query_unknown


def load_projection(path):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    return model(x).cpu().numpy()


def eval_similarity_matrix(sims, asins, ground_truth, query_asins=None):
    """ground_truth: dict asin -> set of related asins. query_asins: if given,
    restrict evaluation to only these queries (used for the cross-type-only
    view, where queries with unknown own type are excluded)."""
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    query_list = query_asins if query_asins is not None else asins
    n = len(query_list)

    sims = sims.copy()
    np.fill_diagonal(sims, -np.inf)
    max_k = max(K_VALUES)

    hits_at = {k: 0 for k in K_VALUES}
    precision_sum = {k: 0.0 for k in K_VALUES}
    for q_asin in query_list:
        i = asin_to_idx[q_asin]
        related = ground_truth.get(q_asin, set())
        row = sims[i]
        top_idx = np.argsort(-row)[:max_k]
        retrieved_all = [asins[j] for j in top_idx]
        for k in K_VALUES:
            retrieved_k = retrieved_all[:k]
            n_hits = sum(1 for r in retrieved_k if r in related)
            if n_hits > 0:
                hits_at[k] += 1
            precision_sum[k] += n_hits / k

    return {
        "n": n,
        **{f"hit_rate@{k}": hits_at[k] / n for k in K_VALUES},
        **{f"precision@{k}": precision_sum[k] / n for k in K_VALUES},
    }


def main():
    asins, raw_embeddings, also_buy, types = load_eval_data()
    print(f"Loaded {len(asins)} eval products (untouched phase 1b/1c sample).")

    n_known_type = sum(1 for a in asins if types.get(a) is not None)
    print(f"Eval products with known type at index {TYPE_INDEX}: {n_known_type}/{len(asins)}")

    cross_type_gt, n_query_unknown = build_cross_type_ground_truth(asins, also_buy, types)
    cross_type_queries = [a for a in asins if types.get(a) is not None]
    n_cross_edges = sum(len(v) for v in cross_type_gt.values())
    print(f"Cross-type-only ground truth: {n_cross_edges} edges across {len(cross_type_queries)} queries "
          f"({n_query_unknown} queries excluded, unknown own type).")

    raw_sims = raw_embeddings @ raw_embeddings.T
    raw_full = eval_similarity_matrix(raw_sims, asins, also_buy)
    for key, expected in EXPECTED_BASELINE.items():
        actual = raw_full[key]
        if abs(actual - expected) > BASELINE_TOLERANCE:
            raise RuntimeError(
                f"SANITY CHECK FAILED: recomputed raw SigLIP {key}={actual:.4f} != expected {expected:.4f}."
            )
    print("Sanity check PASSED: recomputed raw SigLIP metrics match the existing reported baseline.")

    configs = [
        ("Raw SigLIP (baseline)", None, raw_sims),
        ("Phase 7 Model A (random negs, all also_buy)", PHASE7_DIR / "models" / "model_a_random_negs.pt", None),
        ("Phase 7 Model B (hard negs, all also_buy)", PHASE7_DIR / "models" / "model_b_hard_negs.pt", None),
        ("Phase 8 Model A (random negs, heterogeneous-only)", BASE_DIR / "models" / "model_a_random_negs.pt", None),
        ("Phase 8 Model B (hard negs, heterogeneous-only)", BASE_DIR / "models" / "model_b_hard_negs.pt", None),
    ]

    rows = []
    for label, model_path, precomputed_sims in configs:
        if precomputed_sims is not None:
            sims = precomputed_sims
        else:
            model = load_projection(model_path)
            z = project(model, raw_embeddings)
            sims = z @ z.T

        full_metrics = eval_similarity_matrix(sims, asins, also_buy)
        cross_metrics = eval_similarity_matrix(sims, asins, cross_type_gt, query_asins=cross_type_queries)
        rows.append((label, full_metrics, cross_metrics))
        print(f"{label}: FULL {full_metrics} | CROSS-TYPE-ONLY {cross_metrics}")

    lines = [
        "# Phase 8, Step 4: Evaluation on the Untouched 1,872-Product Eval Sample",
        "",
        "Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline "
        "within tolerance -- confirms the untouched eval sample and eval methodology are "
        "unchanged before trusting the learned-model rows below.",
        "",
        f"Cross-type-only ground truth: {n_cross_edges} edges across {len(cross_type_queries)} "
        f"queries ({n_query_unknown} queries excluded from this view, unknown own type at "
        f"index {TYPE_INDEX}).",
        "",
        "## View 1: Full also_buy ground truth (directly comparable to phase 7 and raw SigLIP)",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, full_m, _ in rows:
        lines.append(f"| {label} | {full_m['hit_rate@5']:.3f} | {full_m['hit_rate@10']:.3f} "
                      f"| {full_m['precision@5']:.3f} | {full_m['precision@10']:.3f} |")

    lines += [
        "",
        f"## View 2: Cross-type-only ground truth (N={len(cross_type_queries)} queries, "
        f"{n_cross_edges} cross-type edges -- isolates whether the fix works for what it was trained to predict)",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, _, cross_m in rows:
        lines.append(f"| {label} | {cross_m['hit_rate@5']:.3f} | {cross_m['hit_rate@10']:.3f} "
                      f"| {cross_m['precision@5']:.3f} | {cross_m['precision@10']:.3f} |")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")


if __name__ == "__main__":
    main()

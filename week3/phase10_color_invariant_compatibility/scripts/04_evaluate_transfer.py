"""
Phase 10, step 4.2: transfer test on the untouched Amazon eval sample --
the central test of this phase's hypothesis. Reuses phase 8/9's exact
evaluation mechanism unchanged (same sanity-check gate against the existing
0.502/0.578/0.191/0.144 baseline, same cross-type-only ground truth
construction, N=1,732 queries/1,691 edges). Only this phase's color-
invariant checkpoint is new; applied directly to the existing
week2/phase1b_category_balanced/embeddings/siglip_base.npz -- no new
Amazon images, no re-extraction.

Reports side by side with raw SigLIP, phase 8's Amazon-trained Model A, and
phase 9's naive Polyvore-trained Model A (all carried forward verbatim from
their own results_table.md files) for a direct, complete comparison.
Appends to results_table.md (does not overwrite step 4.1's table).
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
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"
RESULTS_MD = BASE_DIR / "results_table.md"

TYPE_INDEX = 3
EXPECTED_BASELINE = {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144}
BASELINE_TOLERANCE = 0.001
K_VALUES = [5, 10]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# carried forward verbatim from phase 8/9's own results_table.md, not recomputed
PRIOR_NUMBERS = {
    "full": {
        "Raw SigLIP (baseline)": {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144},
        "Phase 8 Model A (Amazon-trained, random negs)": {"hit_rate@5": 0.441, "hit_rate@10": 0.523, "precision@5": 0.160, "precision@10": 0.120},
        "Phase 9 Model A (Polyvore-trained, naive, random negs)": {"hit_rate@5": 0.315, "hit_rate@10": 0.384, "precision@5": 0.098, "precision@10": 0.073},
    },
    "cross_type": {
        "Raw SigLIP (baseline)": {"hit_rate@5": 0.076, "hit_rate@10": 0.112, "precision@5": 0.019, "precision@10": 0.016},
        "Phase 8 Model A (Amazon-trained, random negs)": {"hit_rate@5": 0.078, "hit_rate@10": 0.107, "precision@5": 0.020, "precision@10": 0.016},
        "Phase 9 Model A (Polyvore-trained, naive, random negs)": {"hit_rate@5": 0.046, "hit_rate@10": 0.074, "precision@5": 0.012, "precision@10": 0.011},
    },
}


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


def fmt(v):
    return f"{v:.3f}" if v is not None else "n/a"


def main():
    asins, raw_embeddings, also_buy, types = load_eval_data()
    print(f"Loaded {len(asins)} eval products (untouched phase 1b/1c sample).")

    cross_type_gt, n_query_unknown = build_cross_type_ground_truth(asins, also_buy, types)
    cross_type_queries = [a for a in asins if types.get(a) is not None]
    n_cross_edges = sum(len(v) for v in cross_type_gt.values())

    raw_sims = raw_embeddings @ raw_embeddings.T
    raw_full = eval_similarity_matrix(raw_sims, asins, also_buy)
    for key, expected in EXPECTED_BASELINE.items():
        if abs(raw_full[key] - expected) > BASELINE_TOLERANCE:
            raise RuntimeError(
                f"SANITY CHECK FAILED: recomputed raw SigLIP {key}={raw_full[key]:.4f} != expected {expected:.4f}."
            )
    print("Sanity check PASSED: recomputed raw SigLIP metrics match the existing reported baseline.")

    model = load_projection(BASE_DIR / "models" / "model_color_invariant.pt")
    z = project(model, raw_embeddings)
    sims = z @ z.T
    full_m = eval_similarity_matrix(sims, asins, also_buy)
    cross_m = eval_similarity_matrix(sims, asins, cross_type_gt, query_asins=cross_type_queries)
    label = "Phase 10 color-invariant model"
    print(f"{label}: FULL {full_m} | CROSS-TYPE-ONLY {cross_m}")

    lines = [
        "",
        "# Phase 10, Step 4.2: Transfer Test on the Untouched Amazon Eval Sample",
        "",
        "Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline "
        "within tolerance -- confirms the untouched Amazon eval sample and eval methodology "
        "are unchanged before trusting the transfer-test row below.",
        "",
        f"Cross-type-only ground truth: {n_cross_edges} edges across {len(cross_type_queries)} "
        f"queries ({n_query_unknown} excluded, unknown own type) -- identical to phase 8/9's view.",
        "",
        "## View 1: Full also_buy ground truth",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for lbl, m in PRIOR_NUMBERS["full"].items():
        lines.append(f"| {lbl} | {fmt(m['hit_rate@5'])} | {fmt(m['hit_rate@10'])} "
                      f"| {fmt(m['precision@5'])} | {fmt(m['precision@10'])} |")
    lines.append(f"| **{label}** | **{full_m['hit_rate@5']:.3f}** | **{full_m['hit_rate@10']:.3f}** "
                  f"| **{full_m['precision@5']:.3f}** | **{full_m['precision@10']:.3f}** |")

    lines += [
        "",
        "## View 2: Cross-type-only ground truth (the direct head-to-head)",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for lbl, m in PRIOR_NUMBERS["cross_type"].items():
        lines.append(f"| {lbl} | {fmt(m['hit_rate@5'])} | {fmt(m['hit_rate@10'])} "
                      f"| {fmt(m['precision@5'])} | {fmt(m['precision@10'])} |")
    lines.append(f"| **{label}** | **{cross_m['hit_rate@5']:.3f}** | **{cross_m['hit_rate@10']:.3f}** "
                  f"| **{cross_m['precision@5']:.3f}** | **{cross_m['precision@10']:.3f}** |")
    lines.append("")

    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended transfer-test results to {RESULTS_MD}")


if __name__ == "__main__":
    main()

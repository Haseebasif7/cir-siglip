"""
Phase 9, step 5.2: transfer test on the untouched Amazon eval sample.

Reuses phase 8's exact evaluation mechanism (week3/phase8_cross_category_
retraining/scripts/04_evaluate_projection.py) unchanged: same sanity-check
gate (recomputed raw SigLIP must match the existing 0.502/0.578/0.191/0.144
baseline within 0.001 tolerance before trusting anything else), same
cross-type-only ground truth construction (N=1,732 queries, 1,691 edges,
140 excluded -- unknown own type at categories-breadcrumb index 3). Only
the model checkpoint paths change, swapped to THIS phase's Polyvore-trained
Model A/B -- no new Amazon images, no re-extraction, applied directly to
the existing week2/phase1b_category_balanced/embeddings/siglip_base.npz.

Reports both views (full + cross-type-only) side by side with phase 8's own
Amazon-trained numbers, for a direct head-to-head: does compatibility
learned from clean outfit data (Polyvore) transfer to a completely
different image domain (Amazon product photography) better than
compatibility learned from Amazon's own noisy co-purchase data.

Appends to results_table.md (does not overwrite step 5.1's official-eval table).
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
PHASE8_DIR = BASE_DIR.parent / "phase8_cross_category_retraining"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"
RESULTS_MD = BASE_DIR / "results_table.md"

TYPE_INDEX = 3
EXPECTED_BASELINE = {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144}
BASELINE_TOLERANCE = 0.001
K_VALUES = [5, 10]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# phase 8's own numbers, for direct side-by-side comparison (not recomputed --
# carried forward verbatim from week3/phase8_cross_category_retraining/results_table.md)
PHASE8_NUMBERS = {
    "full": {
        "Raw SigLIP (baseline)": {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144},
        "Phase 8 Model A (Amazon-trained, random negs)": {"hit_rate@5": 0.441, "hit_rate@10": 0.523, "precision@5": 0.160, "precision@10": 0.120},
        "Phase 8 Model B (Amazon-trained, hard negs)": {"hit_rate@5": 0.362, "hit_rate@10": 0.440, "precision@5": 0.117, "precision@10": 0.089},
    },
    "cross_type": {
        "Raw SigLIP (baseline)": {"hit_rate@5": 0.076, "hit_rate@10": 0.112, "precision@5": 0.019, "precision@10": 0.016},
        "Phase 8 Model A (Amazon-trained, random negs)": {"hit_rate@5": 0.078, "hit_rate@10": 0.107, "precision@5": 0.020, "precision@10": 0.016},
        "Phase 8 Model B (Amazon-trained, hard negs)": {"hit_rate@5": 0.059, "hit_rate@10": 0.094, "precision@5": 0.014, "precision@10": 0.012},
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

    configs = [
        ("Raw SigLIP (baseline)", raw_sims),
        ("Phase 9 Model A (Polyvore-trained, random negs)", None),
        ("Phase 9 Model B (Polyvore-trained, hard negs)", None),
    ]
    model_paths = {
        "Phase 9 Model A (Polyvore-trained, random negs)": BASE_DIR / "models" / "model_a_random_negs.pt",
        "Phase 9 Model B (Polyvore-trained, hard negs)": BASE_DIR / "models" / "model_b_hard_negs.pt",
    }

    rows = []
    for label, precomputed in configs:
        if precomputed is not None:
            sims = precomputed
        else:
            model = load_projection(model_paths[label])
            z = project(model, raw_embeddings)
            sims = z @ z.T
        full_m = eval_similarity_matrix(sims, asins, also_buy)
        cross_m = eval_similarity_matrix(sims, asins, cross_type_gt, query_asins=cross_type_queries)
        rows.append((label, full_m, cross_m))
        print(f"{label}: FULL {full_m} | CROSS-TYPE-ONLY {cross_m}")

    lines = [
        "",
        "# Phase 9, Step 5.2: Transfer Test on the Untouched Amazon Eval Sample",
        "",
        "Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline "
        "within tolerance -- confirms the untouched Amazon eval sample and eval methodology "
        "are unchanged before trusting the transfer-test rows below.",
        "",
        f"Cross-type-only ground truth: {n_cross_edges} edges across {len(cross_type_queries)} "
        f"queries ({n_query_unknown} excluded, unknown own type) -- identical to phase 8's view.",
        "",
        "## View 1: Full also_buy ground truth",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, m in PHASE8_NUMBERS["full"].items():
        lines.append(f"| {label} | {m['hit_rate@5']:.3f} | {m['hit_rate@10']:.3f} "
                      f"| {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    for label, full_m, _ in rows[1:]:  # skip raw SigLIP, already in phase8 numbers above
        lines.append(f"| {label} | {full_m['hit_rate@5']:.3f} | {full_m['hit_rate@10']:.3f} "
                      f"| {full_m['precision@5']:.3f} | {full_m['precision@10']:.3f} |")

    lines += [
        "",
        "## View 2: Cross-type-only ground truth (the direct head-to-head)",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, m in PHASE8_NUMBERS["cross_type"].items():
        lines.append(f"| {label} | {m['hit_rate@5']:.3f} | {m['hit_rate@10']:.3f} "
                      f"| {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    for label, _, cross_m in rows[1:]:
        lines.append(f"| {label} | {cross_m['hit_rate@5']:.3f} | {cross_m['hit_rate@10']:.3f} "
                      f"| {cross_m['precision@5']:.3f} | {cross_m['precision@10']:.3f} |")
    lines.append("")

    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended transfer-test results to {RESULTS_MD}")


if __name__ == "__main__":
    main()

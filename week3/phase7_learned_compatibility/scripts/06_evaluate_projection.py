"""
Phase 7, step 6: evaluate on the fully untouched phase 1b/1c eval sample
(1,872 products). Loads ONLY the existing siglip_base.npz and sample_data.csv
from week2/phase1b_category_balanced -- never re-extracted, never
re-downloaded, never touched by anything in this phase's training pool.

Same retrieval-eval methodology used throughout the project
(week2/phase1b_category_balanced/scripts/03_retrieval_eval.py): full N x N
cosine similarity matrix, diagonal set to -inf (self-similarity masked),
top-10 retrieved via argsort, Hit Rate@K / Precision@K against also_buy
ground truth (also_viewed is empty in this dataset, confirmed phase 5 --
also_buy is the entirety of ground truth here).

Mandatory sanity check: the recomputed raw-SigLIP metrics MUST reproduce the
existing reported baseline (HR@5=0.502, HR@10=0.578, P@5=0.191, P@10=0.144)
before anything else in this table is trusted.
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
MODELS_DIR = BASE_DIR / "models"
RESULTS_MD = BASE_DIR / "results_table.md"

EXPECTED_BASELINE = {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144}
BASELINE_TOLERANCE = 0.001

K_VALUES = [5, 10]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_eval_data():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    also_buy = {}
    for _, row in df.iterrows():
        also_buy[row["asin"]] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()

    data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return asins, embeddings, also_buy


def load_projection(name):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    z = model(x).cpu().numpy()
    return z


def eval_similarity_matrix(sims, asins, also_buy):
    n = len(asins)
    sims = sims.copy()
    np.fill_diagonal(sims, -np.inf)
    max_k = max(K_VALUES)
    top_idx = np.argsort(-sims, axis=1)[:, :max_k]

    hits_at = {k: 0 for k in K_VALUES}
    precision_sum = {k: 0.0 for k in K_VALUES}
    for i in range(n):
        related = also_buy.get(asins[i], set())
        retrieved_all = [asins[j] for j in top_idx[i]]
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


def main():
    asins, raw_embeddings, also_buy = load_eval_data()
    print(f"Loaded {len(asins)} eval products (untouched phase 1b/1c sample).")

    raw_sims = raw_embeddings @ raw_embeddings.T
    raw_metrics = eval_similarity_matrix(raw_sims, asins, also_buy)
    print(f"Raw SigLIP (recomputed): {raw_metrics}")

    for key, expected in EXPECTED_BASELINE.items():
        actual = raw_metrics[key]
        if abs(actual - expected) > BASELINE_TOLERANCE:
            raise RuntimeError(
                f"SANITY CHECK FAILED: recomputed raw SigLIP {key}={actual:.4f} does not match "
                f"expected baseline {expected:.4f} (tolerance {BASELINE_TOLERANCE}). "
                f"Something has drifted (embeddings or eval logic) -- halting rather than "
                f"reporting numbers next to a stale baseline row."
            )
    print("Sanity check PASSED: recomputed raw SigLIP metrics match the existing reported baseline.")

    model_a = load_projection("model_a_random_negs")
    model_b = load_projection("model_b_hard_negs")

    z_a = project(model_a, raw_embeddings)
    z_b = project(model_b, raw_embeddings)
    modelA_sims = z_a @ z_a.T
    modelB_sims = z_b @ z_b.T

    modelA_metrics = eval_similarity_matrix(modelA_sims, asins, also_buy)
    modelB_metrics = eval_similarity_matrix(modelB_sims, asins, also_buy)
    print(f"Model A (random negs only): {modelA_metrics}")
    print(f"Model B (random + hard negs): {modelB_metrics}")

    blend_results = {}
    for alpha in [0.7, 0.5, 0.3]:
        blended_sims = alpha * raw_sims + (1 - alpha) * modelB_sims
        blend_results[alpha] = eval_similarity_matrix(blended_sims, asins, also_buy)
        print(f"Blended (alpha={alpha}, raw-weight): {blend_results[alpha]}")

    rows = [
        ("Raw SigLIP (baseline, carried forward)", raw_metrics),
        ("Model A (random negatives only)", modelA_metrics),
        ("Model B (random + hard negatives)", modelB_metrics),
        ("Blended alpha=0.7 (raw-heavy)", blend_results[0.7]),
        ("Blended alpha=0.5 (even)", blend_results[0.5]),
        ("Blended alpha=0.3 (Model-B-heavy)", blend_results[0.3]),
    ]

    lines = [
        "# Phase 7, Step 6: Evaluation on the Untouched 1,872-Product Eval Sample",
        "",
        "Sanity check: recomputed raw SigLIP metrics matched the existing reported "
        "baseline within tolerance (see phase7_notes.md for the exact values) -- "
        "confirms the untouched eval sample and eval methodology are unchanged from "
        "prior phases before trusting the learned/blended rows below.",
        "",
        "| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|",
    ]
    for label, m in rows:
        lines.append(f"| {label} | {m['hit_rate@5']:.3f} | {m['hit_rate@10']:.3f} "
                      f"| {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")


if __name__ == "__main__":
    main()

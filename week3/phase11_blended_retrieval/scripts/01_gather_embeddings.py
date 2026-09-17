"""
Phase 11, step 1: gather the three embedding sets this phase blends, on the
same untouched 1,872-product Amazon eval sample used since phase 1b. No new
training or image processing -- everything already exists:

1. Raw SigLIP embeddings (week2/phase1b_category_balanced/embeddings/siglip_base.npz)
2. Phase 8 Model A's projected embeddings (Amazon-trained, cross-category-only
   compatibility signal) -- computed by running the existing checkpoint's
   frozen forward pass over the same raw SigLIP embeddings (cheap, no
   training), exactly as phase 8/9's own evaluation scripts already do.
3. Phase 9 Model A's projected embeddings (Polyvore-trained compatibility
   signal, applied to Amazon via the same transfer setup phase 9 used).

A sanity check (recomputed raw SigLIP Hit Rate@5/10, Precision@5/10 against
the existing reported baseline) gates saving the bundle, so every downstream
phase 11 script can trust this file without re-deriving it.
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
PHASE9_DIR = BASE_DIR.parent / "phase9_polyvore_compatibility"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"

OUT_NPZ = BASE_DIR / "data" / "embeddings_bundle.npz"

TYPE_INDEX = 3
EXPECTED_BASELINE = {"hit_rate@5": 0.502, "hit_rate@10": 0.578, "precision@5": 0.191, "precision@10": 0.144}
BASELINE_TOLERANCE = 0.001
K_VALUES = [5, 10]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_projection(path):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    return model(x).cpu().numpy()


def eval_similarity_matrix(sims, asins, ground_truth):
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    n = len(asins)
    sims = sims.copy()
    np.fill_diagonal(sims, -np.inf)
    max_k = max(K_VALUES)

    hits_at = {k: 0 for k in K_VALUES}
    precision_sum = {k: 0.0 for k in K_VALUES}
    for q_asin in asins:
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
        **{f"hit_rate@{k}": hits_at[k] / n for k in K_VALUES},
        **{f"precision@{k}": precision_sum[k] / n for k in K_VALUES},
    }


def main():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    also_buy = {}
    for _, row in df.iterrows():
        also_buy[row["asin"]] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()

    data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    raw = data["embeddings"]
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    print(f"Loaded {len(asins)} eval products (untouched phase 1b sample).")

    raw_sims = raw @ raw.T
    raw_metrics = eval_similarity_matrix(raw_sims, asins, also_buy)
    for key, expected in EXPECTED_BASELINE.items():
        actual = raw_metrics[key]
        if abs(actual - expected) > BASELINE_TOLERANCE:
            raise RuntimeError(
                f"SANITY CHECK FAILED: recomputed raw SigLIP {key}={actual:.4f} != expected {expected:.4f}."
            )
    print("Sanity check PASSED: recomputed raw SigLIP metrics match the existing reported baseline.")

    p8_model = load_projection(PHASE8_DIR / "models" / "model_a_random_negs.pt")
    p8_proj = project(p8_model, raw)
    print(f"Phase 8 Model A projected: shape={p8_proj.shape}, "
          f"row-norm mean={np.linalg.norm(p8_proj, axis=1).mean():.4f} (should be ~1.0, ProjectionHead L2-normalizes internally)")

    p9_model = load_projection(PHASE9_DIR / "models" / "model_a_random_negs.pt")
    p9_proj = project(p9_model, raw)
    print(f"Phase 9 Model A projected: shape={p9_proj.shape}, "
          f"row-norm mean={np.linalg.norm(p9_proj, axis=1).mean():.4f} (should be ~1.0, ProjectionHead L2-normalizes internally)")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        asins=np.array(asins, dtype=object),
        raw=raw.astype(np.float32),
        phase8=p8_proj.astype(np.float32),
        phase9=p9_proj.astype(np.float32),
    )
    print(f"\nSaved {OUT_NPZ}")


if __name__ == "__main__":
    main()

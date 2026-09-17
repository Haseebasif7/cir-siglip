"""
Phase 12b, step 1 (part 1): give substitute mode a direct, global, per-item
target in the same 128-d space its embeddings live in.

Phase 12's substitute loss compared PAIRWISE similarity matrices within each
training batch (a relational, batch-local objective) -- diagnosed as too weak
and too indirect. The fix's brief asks for each item's substitute-mode
embedding to directly match "that item's own raw SigLIP embedding" via
cosine similarity. Raw SigLIP is 768-d; z_sub (the shared projection's
output) is 128-d -- cosine similarity is undefined between vectors of
different dimensionality, so a literal reading isn't directly implementable
without picking some way to make the two comparable.

Resolution used here: a FIXED (not learned, not batch-dependent) PCA
projection of raw SigLIP down to 128-d, computed once from the full
251,008-item catalog via eigendecomposition of the 768x768 covariance matrix
(no sklearn, consistent with this project's convention -- see phase 9's AUC
computation). This keeps the fix's actual intent intact: every item gets ONE
fixed target vector representing its own raw SigLIP identity (not something
that depends on which other items happen to be in the current batch), and
substitute loss becomes a direct per-item regression (1 - cosine similarity)
against that fixed target, exactly the "global anchor" the brief asks for,
just made dimensionally well-defined. PCA is unsupervised on frozen
embeddings only -- no outfit/label information touches this step, so there is
no leakage concern from using all 251,008 items (identical in spirit to
phase 9's own use of frozen SigLIP embeddings for every item regardless of
split).
"""
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
OUT_NPZ = BASE_DIR / "data" / "raw_pca128_targets.npz"
REPORT_MD = BASE_DIR / "data" / "pca_variance_check.md"

TARGET_DIM = 128


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    raw = data["embeddings"].astype(np.float64)  # float64 for a stable eigendecomposition
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    print(f"Loaded {raw.shape[0]} raw SigLIP embeddings, dim={raw.shape[1]}.")

    mean_vec = raw.mean(axis=0)
    centered = raw - mean_vec

    cov = (centered.T @ centered) / centered.shape[0]  # (768, 768), cheap
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

    total_var = eigvals.sum()
    top_eigvals = eigvals[-TARGET_DIM:][::-1]
    top_eigvecs = eigvecs[:, -TARGET_DIM:][:, ::-1]  # (768, 128), descending eigenvalue order
    explained = top_eigvals.sum() / total_var

    projected = centered @ top_eigvecs  # (N, 128)
    projected = projected / np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected.astype(np.float32)

    np.savez(OUT_NPZ, item_ids=np.array(item_ids), targets=projected,
              mean_vec=mean_vec.astype(np.float32), projection=top_eigvecs.astype(np.float32))
    print(f"Saved {OUT_NPZ}: {projected.shape}, {explained*100:.1f}% variance explained by top-{TARGET_DIM}.")

    lines = [
        "# Phase 12b, Step 1: PCA-128 Target Variance Check",
        "",
        f"Fixed PCA projection of raw SigLIP (768-d, L2-normalized) down to {TARGET_DIM}-d, "
        "computed once via eigendecomposition of the 768x768 covariance matrix over all "
        f"{raw.shape[0]} Polyvore items (unsupervised, no outfit/label information used).",
        "",
        f"- **{explained*100:.2f}% of total variance explained by the top-{TARGET_DIM} components.**",
        f"- Top-5 component variance shares: {[f'{v/total_var*100:.1f}%' for v in top_eigvals[:5]]}",
        "",
        "This is the target substitute mode's 128-d projection is trained to match directly "
        "(1 - cosine similarity). It is not a lossless copy of raw SigLIP (some variance is "
        "necessarily discarded going from 768-d to 128-d), but it preserves the dominant "
        "structure of the raw embedding space and, critically, gives every item one fixed, "
        "global target vector -- unlike phase 12's batch-local pairwise-similarity objective, "
        "this target does not depend on which other items happen to share a training batch.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()

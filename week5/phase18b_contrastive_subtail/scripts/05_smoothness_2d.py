"""Phase 18b, step 5 (smoothness half): same method as phase 18's
`06_smoothness_2d.py`, re-run on this phase's grid cache for direct
before/after comparison."""
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_NPZ = BASE_DIR / "data" / "grid_topk_cache.npz"
OUT_MD = BASE_DIR / "smoothness_2d.md"

K_EVAL = 10

PHASE18_GAPS = {"axis1": 0.3476, "axis2": 0.2825, "diagonal": 0.2895}


def overlap(a, b):
    return len(set(a[:K_EVAL]) & set(b[:K_EVAL])) / K_EVAL


def main():
    d = np.load(CACHE_NPZ, allow_pickle=True)
    grid = list(d["grid"])
    topk = d["topk_asins"]
    n_queries = topk.shape[2]
    n = len(grid)
    print(f"Loaded grid cache: {n}x{n} grid, {n_queries} queries")

    def mean_overlap(i1a, i2a, i1b, i2b):
        vals = [overlap(topk[i1a, i2a, qi], topk[i1b, i2b, qi]) for qi in range(n_queries)]
        return float(np.mean(vals))

    axis1_adjacent, axis1_distant = [], []
    for i2 in range(n):
        for i1 in range(n - 1):
            axis1_adjacent.append(mean_overlap(i1, i2, i1 + 1, i2))
        axis1_distant.append(mean_overlap(0, i2, n - 1, i2))
    axis1_adj_mean = float(np.mean(axis1_adjacent))
    axis1_dist_mean = float(np.mean(axis1_distant))
    axis1_gap = axis1_adj_mean - axis1_dist_mean

    axis2_adjacent, axis2_distant = [], []
    for i1 in range(n):
        for i2 in range(n - 1):
            axis2_adjacent.append(mean_overlap(i1, i2, i1, i2 + 1))
        axis2_distant.append(mean_overlap(i1, 0, i1, n - 1))
    axis2_adj_mean = float(np.mean(axis2_adjacent))
    axis2_dist_mean = float(np.mean(axis2_distant))
    axis2_gap = axis2_adj_mean - axis2_dist_mean

    diag_adjacent, diag_distant = [], []
    for i in range(n - 1):
        diag_adjacent.append(mean_overlap(i, i, i + 1, i + 1))
    diag_distant.append(mean_overlap(0, 0, n - 1, n - 1))
    anti_adjacent, anti_distant = [], []
    for i in range(n - 1):
        anti_adjacent.append(mean_overlap(i, n - 1 - i, i + 1, n - 2 - i))
    anti_distant.append(mean_overlap(0, n - 1, n - 1, 0))
    diag_adj_mean = float(np.mean(diag_adjacent + anti_adjacent))
    diag_dist_mean = float(np.mean(diag_distant + anti_distant))
    diag_gap = diag_adj_mean - diag_dist_mean

    lines = [
        "# Phase 18b, Step 5: Smoothness Extended to Two Dimensions (Re-Run)\n",
        "Same method as phase 18's `06_smoothness_2d.py`.\n",
        "## Axis 1 (substitute <-> complement), alpha2 held fixed\n",
        f"- Adjacent (|delta alpha1|=0.25): mean overlap = {axis1_adj_mean:.4f}",
        f"- Distant (|delta alpha1|=1.0): mean overlap = {axis1_dist_mean:.4f}",
        f"- **Gap = {axis1_gap:.4f}** (phase 18: {PHASE18_GAPS['axis1']:.4f})\n",
        "## Axis 2 (relevance <-> tail-exposure), alpha1 held fixed\n",
        f"- Adjacent (|delta alpha2|=0.25): mean overlap = {axis2_adj_mean:.4f}",
        f"- Distant (|delta alpha2|=1.0): mean overlap = {axis2_dist_mean:.4f}",
        f"- **Gap = {axis2_gap:.4f}** (phase 18: {PHASE18_GAPS['axis2']:.4f})\n",
        "## Diagonal (both alpha1 and alpha2 move together)\n",
        f"- Adjacent: mean overlap = {diag_adj_mean:.4f}",
        f"- Distant: mean overlap = {diag_dist_mean:.4f}",
        f"- **Gap = {diag_gap:.4f}** (phase 18: {PHASE18_GAPS['diagonal']:.4f})\n",
        "## Comparison\n",
        "| Mechanism | Gap |",
        "|---|---|",
        "| Phase 18, axis 1 | 0.3476 |",
        "| Phase 18, axis 2 | 0.2825 |",
        "| Phase 18, diagonal | 0.2895 |",
        f"| **Phase 18b, axis 1** | **{axis1_gap:.4f}** |",
        f"| **Phase 18b, axis 2** | **{axis2_gap:.4f}** |",
        f"| **Phase 18b, diagonal** | **{diag_gap:.4f}** |",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Saved {OUT_MD}")
    print(f"Axis 1 gap: {axis1_gap:.4f}, Axis 2 gap: {axis2_gap:.4f}, Diagonal gap: {diag_gap:.4f}")


if __name__ == "__main__":
    main()

"""Phase 18, step 8: smoothness extended to two dimensions. Same
adjacent-vs-distant top-10 overlap logic every prior phase used (phases
12d/16/16c/16d/17), but checked separately along each axis (holding the
other alpha fixed) and diagonally across the grid, using the 5x5 grid
cached by 05_evaluate_grid.py.
"""
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_NPZ = BASE_DIR / "data" / "grid_topk_cache.npz"
OUT_MD = BASE_DIR / "smoothness_2d.md"

K_EVAL = 10


def overlap(a, b):
    return len(set(a[:K_EVAL]) & set(b[:K_EVAL])) / K_EVAL


def main():
    d = np.load(CACHE_NPZ, allow_pickle=True)
    grid = list(d["grid"])  # [0.0, 0.25, 0.5, 0.75, 1.0]
    topk = d["topk_asins"]  # (5, 5, n_queries, K_CACHE)
    n_queries = topk.shape[2]
    n = len(grid)
    print(f"Loaded grid cache: {n}x{n} grid, {n_queries} queries")

    def mean_overlap(i1a, i2a, i1b, i2b):
        vals = [overlap(topk[i1a, i2a, qi], topk[i1b, i2b, qi]) for qi in range(n_queries)]
        return float(np.mean(vals))

    # ---- Axis 1 only (alpha2 fixed) ----
    axis1_adjacent, axis1_distant = [], []
    for i2 in range(n):
        for i1 in range(n - 1):
            axis1_adjacent.append(mean_overlap(i1, i2, i1 + 1, i2))
        axis1_distant.append(mean_overlap(0, i2, n - 1, i2))
    axis1_adj_mean = float(np.mean(axis1_adjacent))
    axis1_dist_mean = float(np.mean(axis1_distant))
    axis1_gap = axis1_adj_mean - axis1_dist_mean

    # ---- Axis 2 only (alpha1 fixed) ----
    axis2_adjacent, axis2_distant = [], []
    for i1 in range(n):
        for i2 in range(n - 1):
            axis2_adjacent.append(mean_overlap(i1, i2, i1, i2 + 1))
        axis2_distant.append(mean_overlap(i1, 0, i1, n - 1))
    axis2_adj_mean = float(np.mean(axis2_adjacent))
    axis2_dist_mean = float(np.mean(axis2_distant))
    axis2_gap = axis2_adj_mean - axis2_dist_mean

    # ---- Diagonal (both alpha1 and alpha2 move together) ----
    # Main diagonal direction: (0,0)->(1,1). Anti-diagonal: (0,1)->(1,0).
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
        "# Phase 18, Step 8: Smoothness Extended to Two Dimensions\n",
        "Same adjacent-vs-distant top-10 overlap logic as phases 12d/16/16c/16d/17, applied along each axis "
        "(other alpha held fixed, averaged across all 5 values of it) and diagonally across the grid.\n",
        "## Axis 1 (substitute <-> complement), alpha2 held fixed\n",
        f"- Adjacent (|delta alpha1|=0.25): mean overlap = {axis1_adj_mean:.4f}",
        f"- Distant (|delta alpha1|=1.0, full range): mean overlap = {axis1_dist_mean:.4f}",
        f"- **Gap = {axis1_gap:.4f}**\n",
        "## Axis 2 (relevance <-> tail-exposure), alpha1 held fixed\n",
        f"- Adjacent (|delta alpha2|=0.25): mean overlap = {axis2_adj_mean:.4f}",
        f"- Distant (|delta alpha2|=1.0, full range): mean overlap = {axis2_dist_mean:.4f}",
        f"- **Gap = {axis2_gap:.4f}**\n",
        "## Diagonal (both alpha1 and alpha2 move together)\n",
        f"- Adjacent (one 0.25 step along either diagonal): mean overlap = {diag_adj_mean:.4f}",
        f"- Distant (full corner-to-corner, either diagonal): mean overlap = {diag_dist_mean:.4f}",
        f"- **Gap = {diag_gap:.4f}**\n",
        "## Comparison to this project's prior 1D smoothness gaps\n",
        "| Mechanism | Gap |",
        "|---|---|",
        "| Phase 12d (substitute/complement, shared trunk) | 0.4751 |",
        "| Phase 16d (relevance/tail, dedicated capacity) | 0.6110 |",
        "| Phase 17 (substitute/complement, dedicated capacity, Polyvore) | 0.6646 |",
        f"| **Phase 18, axis 1 (substitute/complement, 4-head, Amazon)** | **{axis1_gap:.4f}** |",
        f"| **Phase 18, axis 2 (relevance/tail, 4-head, Amazon)** | **{axis2_gap:.4f}** |",
        f"| **Phase 18, diagonal (both axes together)** | **{diag_gap:.4f}** |",
        "",
        "## Verdict\n",
    ]
    monotonic_note = ("Gaps are computed from a coarser 5-point grid per axis (vs. the usual 11-point 1D "
                       "sweep), so these numbers are directly comparable in spirit but not on an identical "
                       "point density -- flagged here rather than presented as an apples-to-apples match.")
    lines.append(monotonic_note)
    lines.append("")
    if axis1_gap > 0.1 and axis2_gap > 0.1 and diag_gap > 0.1:
        lines.append("**Both individual axes, and the diagonal combining them, show real, substantial "
                      "smoothness gaps -- this is a genuinely, smoothly navigable 2D control space, not just "
                      "two working 1D dials bolted together with a sharp seam between them.**")
    else:
        lines.append("**At least one of the three gaps (axis 1, axis 2, diagonal) is small** -- reported "
                      "plainly; see phase18_notes.md for interpretation alongside the axis-independence "
                      "finding from step 7.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved {OUT_MD}")
    print(f"Axis 1 gap: {axis1_gap:.4f}, Axis 2 gap: {axis2_gap:.4f}, Diagonal gap: {diag_gap:.4f}")


if __name__ == "__main__":
    main()

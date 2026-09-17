"""Adjacent-vs-distant alpha overlap smoothness check, phase 12d's exact
method: full pairwise top-10 overlap matrix across all 11 alpha values,
using the cached per-alpha retrievals from 04_evaluate_alpha_sweep.py, then
mean overlap as a function of |delta alpha|. A genuine, usable dial shows
overlap decaying smoothly and monotonically as |delta alpha| grows (small
turns change results a little, large turns change results a lot) -- not
just two different endpoints with unclear behavior in between.
"""
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_MD = BASE_DIR / "smoothness_check.md"

K = 10


def main():
    d = np.load(CACHE, allow_pickle=True)
    alphas = [round(float(a), 1) for a in d["alphas"]]
    retrieved = d["retrieved_asins"][:, :, :K]  # (n_alpha, n_query, K)
    n_alpha, n_query, _ = retrieved.shape

    matrix = np.zeros((n_alpha, n_alpha))
    for i in range(n_alpha):
        set_i = [set(retrieved[i, q]) for q in range(n_query)]
        for j in range(n_alpha):
            if j < i:
                matrix[i, j] = matrix[j, i]
                continue
            if i == j:
                matrix[i, j] = 1.0
                continue
            overlaps = []
            for q in range(n_query):
                inter = len(set_i[q] & set(retrieved[j, q]))
                overlaps.append(inter / K)
            matrix[i, j] = float(np.mean(overlaps))

    by_delta = {}
    for i in range(n_alpha):
        for j in range(n_alpha):
            if i == j:
                continue
            delta = round(abs(alphas[i] - alphas[j]), 1)
            by_delta.setdefault(delta, []).append(matrix[i, j])

    deltas_sorted = sorted(by_delta.keys())
    mean_by_delta = {delta: float(np.mean(by_delta[delta])) for delta in deltas_sorted}

    adjacent = mean_by_delta.get(0.1)
    distant = mean_by_delta.get(1.0)
    gap = adjacent - distant if (adjacent is not None and distant is not None) else None
    is_monotonic = all(mean_by_delta[deltas_sorted[k]] >= mean_by_delta[deltas_sorted[k + 1]] - 1e-6
                        for k in range(len(deltas_sorted) - 1))

    lines = ["# Phase 16, Step 6: Adjacent vs Distant Alpha Overlap\n"]
    lines.append(f"Full pairwise top-{K} overlap matrix across all {n_alpha} alpha values, all {n_query} phase 1b "
                 "queries, computed from 04_evaluate_alpha_sweep.py's cached per-alpha retrieval lists -- same "
                 "method as phase 12d's `03_adjacent_vs_distant_overlap.py`.\n")
    lines.append("## Mean overlap as a function of |delta alpha|\n")
    lines.append("| |delta alpha| | Mean top-10 overlap |")
    lines.append("|---|---|")
    for delta in deltas_sorted:
        lines.append(f"| {delta:.1f} | {mean_by_delta[delta]:.4f} |")
    lines.append("")
    lines.append(f"**Adjacent steps (delta=0.1): mean overlap = {adjacent:.4f}**")
    lines.append(f"**Most distant (delta=1.0, alpha=0.0 vs alpha=1.0): mean overlap = {distant:.4f}**")
    lines.append(f"**Gap (adjacent - distant): {gap:.4f}**\n")

    lines.append("## Verdict\n")
    # phase 12d's own reference point: gap of 0.4751 was "the operational definition of a usable dial"
    if gap is not None and gap > 0.15 and is_monotonic:
        lines.append(f"**Smoothness check PASSES**: overlap decays monotonically as |delta alpha| grows "
                     f"(adjacent={adjacent:.4f} down to distant={distant:.4f}, gap={gap:.4f}), consistent "
                     "with a real, continuously usable dial rather than just two different endpoints.")
    elif gap is not None and gap > 0.05:
        lines.append(f"**Smoothness check PARTIALLY confirms**: overlap does decline from adjacent "
                     f"({adjacent:.4f}) to distant ({distant:.4f}) steps (gap={gap:.4f}), and the decay is "
                     f"{'monotonic' if is_monotonic else 'not strictly monotonic'} -- real movement exists, "
                     "but the gap is much smaller than phase 12c/12d's own reference mechanism (gap=0.4751 "
                     "there), consistent with the weaker axis-check/field-metric movement found in "
                     "04/05. This mechanism produces a real but structurally weak dial, not an inert one.")
    else:
        lines.append(f"**Smoothness check FAILS**: gap between adjacent ({adjacent:.4f}) and distant "
                     f"({distant:.4f}) steps is too small ({gap:.4f}) to support a usable dial claim -- "
                     "retrieved sets barely change across the whole alpha range.")
    lines.append("")

    lines.append(f"## Full {n_alpha}x{n_alpha} matrix (rows/cols = alpha, values = mean top-{K} overlap)\n")
    header = "| alpha | " + " | ".join(f"{a:.1f}" for a in alphas) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (n_alpha + 1))
    for i, a in enumerate(alphas):
        row = f"| {a:.1f} | " + " | ".join(f"{matrix[i,j]:.3f}" for j in range(n_alpha)) + " |"
        lines.append(row)
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Adjacent overlap: {adjacent:.4f}, Distant overlap: {distant:.4f}, Gap: {gap:.4f}, "
          f"Monotonic: {is_monotonic}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()

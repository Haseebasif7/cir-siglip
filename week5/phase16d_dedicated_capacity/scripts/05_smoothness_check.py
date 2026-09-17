"""Adjacent-vs-distant alpha overlap smoothness check, identical method to
phases 16/16c (phase 12d's method).
"""
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_MD = BASE_DIR / "smoothness_check.md"

K = 10
PHASE16_GAP = 0.0534
PHASE16C_GAP = 0.1261


def main():
    d = np.load(CACHE, allow_pickle=True)
    alphas = [round(float(a), 1) for a in d["alphas"]]
    retrieved = d["retrieved_asins"][:, :, :K]
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
            overlaps = [len(set_i[q] & set(retrieved[j, q])) / K for q in range(n_query)]
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
    gap = adjacent - distant
    is_monotonic = all(mean_by_delta[deltas_sorted[k]] >= mean_by_delta[deltas_sorted[k + 1]] - 1e-6
                        for k in range(len(deltas_sorted) - 1))

    lines = ["# Phase 16d: Adjacent vs Distant Alpha Overlap\n"]
    lines.append(f"Same method as phases 16/16c (phase 12d's method), full pairwise top-{K} overlap matrix "
                 f"across all {n_alpha} alpha values, all {n_query} phase 1b queries.\n")
    lines.append("## Mean overlap as a function of |delta alpha|\n")
    lines.append("| |delta alpha| | Mean top-10 overlap |")
    lines.append("|---|---|")
    for delta in deltas_sorted:
        lines.append(f"| {delta:.1f} | {mean_by_delta[delta]:.4f} |")
    lines.append("")
    lines.append(f"**Adjacent steps (delta=0.1): mean overlap = {adjacent:.4f}**")
    lines.append(f"**Most distant (delta=1.0): mean overlap = {distant:.4f}**")
    lines.append(f"**Gap (adjacent - distant): {gap:.4f}**\n")

    lines.append("## Three-way comparison\n")
    lines.append(f"- Phase 16's gap: {PHASE16_GAP:.4f}")
    lines.append(f"- Phase 16c's gap: {PHASE16C_GAP:.4f}")
    lines.append(f"- Phase 16d's gap: {gap:.4f}\n")

    lines.append("## Verdict\n")
    lines.append(f"Decay is {'monotonic' if is_monotonic else 'NOT strictly monotonic'} -- "
                 + ("consistent with a genuine, continuously usable dial even under the new dedicated-heads "
                    "architecture (confirms blending two independently-normalized, independently-trained "
                    "embeddings still interpolates coherently, the thing this check was specifically built to "
                    "verify rather than assume)."
                    if is_monotonic else
                    "the new architecture's blend-then-renormalize interpolation may not be producing a "
                    "coherent dial -- flagged directly rather than assumed away."))
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
    print(f"Adjacent: {adjacent:.4f}, Distant: {distant:.4f}, Gap: {gap:.4f}, Monotonic: {is_monotonic}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()

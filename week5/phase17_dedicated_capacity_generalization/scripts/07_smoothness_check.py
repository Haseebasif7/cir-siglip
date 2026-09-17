"""
Phase 17, step 5.3: adjacent-vs-distant alpha overlap smoothness check,
identical method to phases 12d/16/16c/16d (phase 12d's original method).
"""
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "data" / "topk_cache.json"
OUT_MD = BASE_DIR / "smoothness_check.md"

ALPHAS = [round(0.1 * i, 1) for i in range(11)]
TOP_K = 10

PHASE12D_GAP = 0.4751
PHASE16_GAP = 0.0534
PHASE16C_GAP = 0.1261
PHASE16D_GAP = 0.6110


def overlap(list_a, list_b):
    return len(set(list_a) & set(list_b)) / TOP_K


def main():
    import json
    with open(CACHE) as f:
        cache = json.load(f)  # {alpha_str: {qi_str: [top10]}}

    common_qis = set(cache[str(ALPHAS[0])].keys())
    for a in ALPHAS[1:]:
        common_qis &= set(cache[str(a)].keys())
    common_qis = sorted(common_qis)
    print(f"{len(common_qis)} queries present at all {len(ALPHAS)} alpha values.")

    n = len(ALPHAS)
    matrix = np.zeros((n, n))
    for i in range(n):
        a_i = cache[str(ALPHAS[i])]
        for j in range(n):
            if j < i:
                matrix[i, j] = matrix[j, i]
                continue
            if i == j:
                matrix[i, j] = 1.0
                continue
            a_j = cache[str(ALPHAS[j])]
            vals = [overlap(a_i[qi], a_j[qi]) for qi in common_qis]
            matrix[i, j] = float(np.mean(vals))

    by_delta = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = round(abs(ALPHAS[i] - ALPHAS[j]), 1)
            by_delta.setdefault(delta, []).append(matrix[i, j])
    deltas_sorted = sorted(by_delta.keys())
    mean_by_delta = {delta: float(np.mean(by_delta[delta])) for delta in deltas_sorted}

    adjacent = mean_by_delta[0.1]
    distant = mean_by_delta[1.0]
    gap = adjacent - distant
    is_monotonic = all(mean_by_delta[deltas_sorted[k]] >= mean_by_delta[deltas_sorted[k + 1]] - 1e-6
                        for k in range(len(deltas_sorted) - 1))

    lines = ["# Phase 17: Adjacent vs Distant Alpha Overlap\n"]
    lines.append(f"Same method as phases 12d/16/16c/16d, full pairwise top-{TOP_K} overlap matrix "
                 f"across all {n} alpha values, {len(common_qis)} queries.\n")
    lines.append("## Mean overlap as a function of |delta alpha|\n")
    lines.append("| |delta alpha| | Mean top-10 overlap |")
    lines.append("|---|---|")
    for delta in deltas_sorted:
        lines.append(f"| {delta:.1f} | {mean_by_delta[delta]:.4f} |")
    lines.append("")
    lines.append(f"**Adjacent steps (delta=0.1): mean overlap = {adjacent:.4f}**")
    lines.append(f"**Most distant (delta=1.0): mean overlap = {distant:.4f}**")
    lines.append(f"**Gap (adjacent - distant): {gap:.4f}**\n")

    lines.append("## Comparison across the project's mechanisms\n")
    lines.append("| Mechanism | Architecture | Gap |")
    lines.append("|---|---|---|")
    lines.append(f"| Phase 12d (substitute/complement) | shared trunk + additive correction | {PHASE12D_GAP:.4f} |")
    lines.append(f"| Phase 16 (relevance/tail) | shared trunk + additive correction | {PHASE16_GAP:.4f} |")
    lines.append(f"| Phase 16c (relevance/tail) | shared trunk + additive correction | {PHASE16C_GAP:.4f} |")
    lines.append(f"| Phase 16d (relevance/tail) | dedicated capacity | {PHASE16D_GAP:.4f} |")
    lines.append(f"| **Phase 17 (substitute/complement)** | **dedicated capacity** | **{gap:.4f}** |")
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(f"Decay is {'monotonic' if is_monotonic else 'NOT strictly monotonic'} -- "
                 + ("consistent with a genuine, continuously usable dial under the dedicated-heads "
                    "architecture, on the substitute/complement axis this time (not just the "
                    "relevance/tail axis phase 16d tested)."
                    if is_monotonic else
                    "the dedicated-heads architecture's blend-then-renormalize interpolation may not "
                    "be producing a coherent dial here -- flagged directly rather than assumed away."))
    lines.append("")

    lines.append(f"## Full {n}x{n} matrix (rows/cols = alpha, values = mean top-{TOP_K} overlap)\n")
    header = "| alpha | " + " | ".join(f"{a:.1f}" for a in ALPHAS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (n + 1))
    for i, a in enumerate(ALPHAS):
        row = f"| {a:.1f} | " + " | ".join(f"{matrix[i,j]:.3f}" for j in range(n)) + " |"
        lines.append(row)
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Adjacent: {adjacent:.4f}, Distant: {distant:.4f}, Gap: {gap:.4f}, Monotonic: {is_monotonic}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()

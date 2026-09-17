"""Adjacent-vs-distant alpha overlap smoothness check, identical method to
phase 16's 06_smoothness_check.py (phase 12d's method), for direct
comparability. This phase's own success bar (step 5 of the brief): the gap
must at least double phase 16's 0.0534.
"""
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_MD = BASE_DIR / "smoothness_check.md"

K = 10
PHASE16_GAP = 0.0534
SUCCESS_BAR_GAP = PHASE16_GAP * 2  # "at least doubling phase 16's gap"


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
    clears_bar = gap >= SUCCESS_BAR_GAP

    lines = ["# Phase 16c, Step 4: Adjacent vs Distant Alpha Overlap\n"]
    lines.append(f"Same method as phase 16's `06_smoothness_check.py` (phase 12d's method), full pairwise "
                 f"top-{K} overlap matrix across all {n_alpha} alpha values, all {n_query} phase 1b queries.\n")
    lines.append("## Mean overlap as a function of |delta alpha|\n")
    lines.append("| |delta alpha| | Mean top-10 overlap |")
    lines.append("|---|---|")
    for delta in deltas_sorted:
        lines.append(f"| {delta:.1f} | {mean_by_delta[delta]:.4f} |")
    lines.append("")
    lines.append(f"**Adjacent steps (delta=0.1): mean overlap = {adjacent:.4f}**")
    lines.append(f"**Most distant (delta=1.0): mean overlap = {distant:.4f}**")
    lines.append(f"**Gap (adjacent - distant): {gap:.4f}**\n")

    lines.append("## Direct comparison to phase 16 and this phase's own success bar\n")
    lines.append(f"- Phase 16's gap: {PHASE16_GAP:.4f}")
    lines.append(f"- Phase 16c's gap: {gap:.4f} ({gap/PHASE16_GAP:.2f}x phase 16's)")
    lines.append(f"- Success bar (this phase's own, set in advance): at least 2x phase 16's gap = "
                 f"{SUCCESS_BAR_GAP:.4f}")
    lines.append(f"- **{'CLEARS the bar' if clears_bar else 'DOES NOT clear the bar'}**\n")

    lines.append("## Verdict\n")
    if clears_bar and is_monotonic:
        lines.append(f"**Smoothness bar CLEARED**: gap ({gap:.4f}) is at least double phase 16's ({PHASE16_GAP:.4f}), "
                     f"and decay is monotonic -- a genuinely stronger, more usable dial than phase 16 produced.")
    else:
        lines.append(f"**Smoothness bar NOT cleared**: gap ({gap:.4f}) does not reach double phase 16's "
                     f"({SUCCESS_BAR_GAP:.4f} required). "
                     + ("Decay is monotonic, real movement exists, just not enough to clear this phase's own "
                        "pre-declared bar." if is_monotonic else "Decay is also not strictly monotonic."))
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
    print(f"Adjacent: {adjacent:.4f}, Distant: {distant:.4f}, Gap: {gap:.4f}, Monotonic: {is_monotonic}, "
          f"Clears bar: {clears_bar}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()

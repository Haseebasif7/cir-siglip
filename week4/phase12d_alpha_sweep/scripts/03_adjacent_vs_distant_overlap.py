"""
Phase 12d, step 3: does retrieval behavior change smoothly at the RETRIEVAL
level (top-10 item sets), not just at the embedding level? Embedding
interpolation being linear/smooth doesn't guarantee ranking is smooth, since
nearest-neighbor retrieval is a nonlinear operation on top of the embeddings.

Uses the per-query top-10 cache from step 1 (`data/topk_cache.json`, same
500-query sample at all 11 alpha values) to build the full 11x11 pairwise
overlap matrix, then summarizes overlap as a function of |delta alpha| --
the real operational test of "a little turn changes a little, a lot of turn
changes a lot."
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
TOPK_CACHE_JSON = BASE_DIR / "data" / "topk_cache.json"
OUT_MD = BASE_DIR / "adjacent_vs_distant_overlap.md"

ALPHAS = [round(0.1 * i, 1) for i in range(11)]
TOP_K = 10


def overlap(list_a, list_b):
    return len(set(list_a) & set(list_b)) / TOP_K


def main():
    with open(TOPK_CACHE_JSON) as f:
        cache = json.load(f)  # {alpha_str: {qi_str: [top10]}}

    # queries present at every alpha (should be ~all 500, guard against any skipped query)
    common_qis = set(cache[str(ALPHAS[0])].keys())
    for a in ALPHAS[1:]:
        common_qis &= set(cache[str(a)].keys())
    common_qis = sorted(common_qis)
    print(f"{len(common_qis)} queries present at all 11 alpha values.")

    n = len(ALPHAS)
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        a_i, a_j = cache[str(ALPHAS[i])], cache[str(ALPHAS[j])]
        vals = [overlap(a_i[qi], a_j[qi]) for qi in common_qis]
        matrix[i, j] = matrix[j, i] = float(np.mean(vals))
    np.fill_diagonal(matrix, 1.0)

    # overlap as a function of |delta alpha|
    by_delta = {}
    for i, j in combinations(range(n), 2):
        delta = round(abs(ALPHAS[i] - ALPHAS[j]), 1)
        by_delta.setdefault(delta, []).append(matrix[i, j])
    delta_means = {d: float(np.mean(vs)) for d, vs in sorted(by_delta.items())}

    adjacent_mean = delta_means[0.1]
    distant_mean = delta_means[max(delta_means)]  # delta=1.0, i.e. alpha=0.0 vs alpha=1.0

    lines = [
        "# Phase 12d, Step 3: Adjacent vs Distant Alpha Overlap",
        "",
        f"Full pairwise top-10 overlap matrix across all 11 alpha values, {len(common_qis)}-query "
        "sample (same 500-query overlap sample as step 1, seed=42), computed from step 1's "
        "cached per-alpha retrieval lists.",
        "",
        "## Mean overlap as a function of |delta alpha| (the real test)",
        "",
        "| |delta alpha| | Mean top-10 overlap |",
        "|---|---|",
    ]
    for d, m in delta_means.items():
        lines.append(f"| {d:.1f} | {m:.4f} |")
    lines.append("")
    lines.append(f"**Adjacent steps (delta=0.1): mean overlap = {adjacent_mean:.4f}**")
    lines.append(f"**Most distant (delta=1.0, alpha=0.0 vs alpha=1.0): mean overlap = {distant_mean:.4f}**")
    lines.append("")

    monotonic_decay = all(
        delta_means[round(d1, 1)] >= delta_means[round(d2, 1)] - 0.02  # small noise tolerance
        for d1, d2 in zip(sorted(delta_means)[:-1], sorted(delta_means)[1:])
    )
    gap = adjacent_mean - distant_mean

    lines.append("## Verdict")
    lines.append("")
    if gap > 0.15 and monotonic_decay:
        lines.append(
            f"**Real, smooth relationship confirmed**: overlap decays consistently as |delta "
            f"alpha| grows (adjacent={adjacent_mean:.4f} down to distant={distant_mean:.4f}, a "
            f"gap of {gap:.4f}), not just a binary near-vs-far difference. This is the "
            "operational definition of a usable dial -- small turns change results a little, "
            "large turns change results a lot."
        )
    elif gap > 0.05:
        lines.append(
            f"**Partial confirmation**: adjacent steps do overlap more than distant ones "
            f"(gap={gap:.4f}), but the decay is not cleanly monotonic across every |delta "
            "alpha| step -- see the full table above. See `../phase12d_notes.md` for what "
            "this means for the overall dial-quality verdict."
        )
    else:
        lines.append(
            f"**Not confirmed**: adjacent-step overlap ({adjacent_mean:.4f}) is not "
            f"meaningfully higher than distant-step overlap ({distant_mean:.4f}, gap="
            f"{gap:.4f}) -- retrieval does not change smoothly with alpha at the item-ranking "
            "level, even if the embeddings interpolate smoothly. See `../phase12d_notes.md`."
        )
    lines.append("")

    lines.append("## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)")
    lines.append("")
    header = "| alpha | " + " | ".join(f"{a:.1f}" for a in ALPHAS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (n + 1))
    for i, a in enumerate(ALPHAS):
        row = " | ".join(f"{matrix[i,j]:.3f}" for j in range(n))
        lines.append(f"| {a:.1f} | {row} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Adjacent mean: {adjacent_mean:.4f}, Distant mean: {distant_mean:.4f}, gap: {gap:.4f}")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

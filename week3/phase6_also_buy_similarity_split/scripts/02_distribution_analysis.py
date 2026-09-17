"""
Phase 6, step 2: examine the distribution of also_buy edge similarities
(SigLIP) computed in step 1. Histogram + KDE plot, summary statistics, and
an explicit bimodality check -- not just eyeballing the plot.

Bimodality is checked two ways, since either one alone can mislead:
1. Peak-counting on the KDE curve (how many local maxima, with a minimum
   prominence so noise-level bumps don't count as "modes").
2. Sarle's bimodality coefficient (BC = (skew^2 + 1) / excess_kurtosis+3),
   a standard descriptive heuristic where BC > 0.555 (the value for a
   uniform distribution) is often used as a rough bimodality signal --
   reported as one more data point, not a definitive test.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
EDGE_JSON = BASE_DIR / "data" / "edge_similarities_siglip.json"
OUT_MD = BASE_DIR / "similarity_distribution.md"
OUT_PLOT = BASE_DIR / "data" / "similarity_distribution_siglip.png"


def count_kde_peaks(sims, min_prominence_frac=0.03):
    kde = stats.gaussian_kde(sims)
    grid = np.linspace(sims.min(), sims.max(), 1000)
    density = kde(grid)
    peak_idx = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peak_idx.append(i)
    # filter peaks by prominence relative to the overall density range, so
    # small wiggles in the KDE don't get counted as separate modes
    d_range = density.max() - density.min()
    min_prom = min_prominence_frac * d_range
    kept = []
    for i in peak_idx:
        left_min = density[:i + 1].min()
        right_min = density[i:].min()
        prominence = density[i] - max(left_min, right_min)
        if prominence >= min_prom:
            kept.append(i)
    return grid, density, [grid[i] for i in kept]


def main(technique="siglip", edge_json=EDGE_JSON, out_md=OUT_MD, out_plot=OUT_PLOT):
    with open(edge_json) as f:
        d = json.load(f)
    sims = np.array([e["similarity"] for e in d["edges"]])
    n = len(sims)

    mean, median, std = sims.mean(), np.median(sims), sims.std()
    skew = stats.skew(sims)
    kurt = stats.kurtosis(sims)  # excess kurtosis (0 = normal)
    bc = (skew ** 2 + 1) / (kurt + 3)  # Sarle's bimodality coefficient

    grid, density, peaks = count_kde_peaks(sims)
    n_modes = len(peaks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(sims, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"{technique.upper()}: also_buy edge similarity (histogram, n={n})")
    axes[0].set_xlabel("cosine similarity")
    axes[0].set_ylabel("count")
    axes[0].axvline(mean, color="red", linestyle="--", linewidth=1, label=f"mean={mean:.3f}")
    axes[0].axvline(median, color="green", linestyle=":", linewidth=1, label=f"median={median:.3f}")
    axes[0].legend(fontsize=8)

    axes[1].plot(grid, density, color="#4C72B0")
    axes[1].fill_between(grid, density, alpha=0.3, color="#4C72B0")
    for p in peaks:
        axes[1].axvline(p, color="darkorange", linestyle="--", linewidth=1)
    axes[1].set_title(f"{technique.upper()}: KDE (detected modes: {n_modes})")
    axes[1].set_xlabel("cosine similarity")
    axes[1].set_ylabel("density")

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved {out_plot}")

    if n_modes >= 2 and bc > 0.555:
        shape_reading = (
            f"**Both checks point toward bimodality**: the KDE has {n_modes} distinct "
            f"prominent peaks (at similarity ~{', ~'.join(f'{p:.3f}' for p in peaks)}), "
            f"and Sarle's bimodality coefficient ({bc:.3f}) exceeds the 0.555 rough "
            "threshold. Worth checking the qualitative examples (step 3) to see if this "
            "numeric pattern corresponds to a real substitute/complement distinction."
        )
    elif n_modes >= 2:
        shape_reading = (
            f"**Mixed signal**: the KDE shows {n_modes} prominent peaks (at similarity "
            f"~{', ~'.join(f'{p:.3f}' for p in peaks)}), suggesting some structure, but "
            f"Sarle's bimodality coefficient ({bc:.3f}) stays below the 0.555 rough "
            "threshold typically associated with bimodality -- the peaks may be real but "
            "modest, not a clean two-group split. Needs the qualitative check to interpret."
        )
    else:
        shape_reading = (
            f"**Unimodal / continuous**: the KDE shows a single dominant peak (no second "
            f"prominent mode detected), and Sarle's bimodality coefficient ({bc:.3f}) stays "
            "below the 0.555 rough threshold. This is more consistent with a single "
            "continuous spread of similarity values than two distinct underlying groups -- "
            "no natural numeric split point is visible from the distribution shape alone."
        )

    lines = [
        f"# Phase 6, Step 2: also_buy Edge Similarity Distribution ({technique.upper()})",
        "",
        f"n = {n} within-sample also_buy edges (directed pairs, phase 1b's 1,872-product "
        "sample, SigLIP embeddings).",
        "",
        "## Summary statistics",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Mean | {mean:.4f} |",
        f"| Median | {median:.4f} |",
        f"| Std dev | {std:.4f} |",
        f"| Min | {sims.min():.4f} |",
        f"| Max | {sims.max():.4f} |",
        f"| Skewness | {skew:.4f} |",
        f"| Excess kurtosis | {kurt:.4f} |",
        f"| Sarle's bimodality coefficient | {bc:.4f} (0.555 = rough bimodality threshold) |",
        "",
        f"## Detected KDE modes: {n_modes}",
        "",
        (f"Peak location(s): {', '.join(f'{p:.4f}' for p in peaks)}" if peaks else "No prominent peak detected beyond the single global maximum."),
        "",
        f"![distribution plot]({out_plot.name})",
        "",
        "## Shape reading",
        "",
        shape_reading,
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_md}")
    print(f"n={n} mean={mean:.4f} median={median:.4f} std={std:.4f} bc={bc:.4f} n_modes={n_modes} peaks={peaks}")


if __name__ == "__main__":
    main()

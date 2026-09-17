"""
Phase 12d, step 5: single line plot, all four diagnostic metrics vs alpha,
so the shape of the relationship (smooth, stepped, erratic) is immediately
visible rather than read out of a table.

Color choices follow the dataviz skill's procedure: four categorical hues
(Okabe-Ito, CVD-safe) assigned in a fixed order, validated with the skill's
own `validate_palette.js` before use (all four checks passed; one contrast
WARN on the orange line against a light surface, mitigated here with a
thicker line + marker + direct end-of-line label rather than relying on hue
alone). Single shared y-axis (all four metrics are already 0-1-ish
similarity/rate/overlap fractions) -- no dual-axis, per the skill's
non-negotiable.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE_DIR / "data" / "alpha_sweep_results.json"
OUT_PNG = BASE_DIR / "plots" / "alpha_sweep.png"

ALPHAS = [round(0.1 * i, 1) for i in range(11)]

# Fixed-order categorical assignment, Okabe-Ito CVD-safe hues, validated via
# the dataviz skill's validate_palette.js (all checks passed).
SERIES = [
    ("axis1_visual_sim", "Visual similarity to query (a)", "#0072B2", "o"),
    ("axis2_hit_rate", "Co-occurrence hit rate (b)", "#E69F00", "s"),
    ("overlap_with_raw", "Overlap with raw SigLIP (c)", "#009E73", "^"),
    ("overlap_with_complement", "Overlap with complement mode (d)", "#D55E00", "D"),
]


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, label, color, marker in SERIES:
        values = [results[str(a)][key] for a in ALPHAS]
        ax.plot(ALPHAS, values, color=color, marker=marker, markersize=7,
                linewidth=2, label=label)
        # direct end-of-line label, mitigates the orange line's contrast WARN
        # against a light surface (per validate_palette.js output)
        ax.annotate(label.split(" (")[0], xy=(1.0, values[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=color,
                    va="center", ha="left")

    ax.set_xlabel("alpha (0.0 = pure complement mode, 1.0 = pure substitute mode)")
    ax.set_ylabel("Metric value")
    ax.set_title("Phase 12d: Alpha Interpolation Sweep -- Four Diagnostic Metrics")
    ax.set_xlim(-0.02, 1.28)
    ax.set_xticks(ALPHAS)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()

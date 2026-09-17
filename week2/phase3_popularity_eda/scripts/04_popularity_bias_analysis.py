"""
Phase 3, Step 4 + 5: popularity bias analysis and visualizations.

For each of the four techniques (ResNet50, CLIP ViT-B/32, FashionCLIP,
SigLIP), computes:
  1. baseline composition       -- tier mix of the 1,872-product sample itself
  2. retrieved composition      -- tier mix of top-5/top-10 retrieved items, pooled
  3. ground truth composition   -- tier mix of every query's real also_buy/
                                    also_viewed targets (not the retrieved items)
  4. ARP (Average Recommendation Popularity) -- mean ref_count of top-5/top-10 items
  5. catalog coverage            -- fraction of the 1,872-product sample that
                                    ever appears in any query's top-5

Baseline and ground-truth composition don't depend on technique (same sample,
same real relatedness edges for all four), so they're computed once and
reused across every technique's row/chart for a fair side-by-side read.

Tiers come from data/popularity_lookup.csv (step 1+2's catalog-wide table,
covering every asin ever referenced anywhere in the metadata file -- so any
retrieved item or ground-truth target, even one outside the 1,872 sample, has
a defined tier).
"""
import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PHASE3_DIR = BASE_DIR.parent
PHASE1B_DIR = PHASE3_DIR.parent / "phase1b_category_balanced"

SAMPLE_CSV = PHASE1B_DIR / "data" / "sample_data.csv"
LOOKUP_CSV = PHASE3_DIR / "data" / "popularity_lookup.csv"
RETRIEVALS_JSON = PHASE3_DIR / "data" / "full_retrievals.json"

RESULTS_MD = PHASE3_DIR / "results_table.md"
BUCKET_MD = PHASE3_DIR / "bucket_composition.md"
PLOTS_DIR = PHASE3_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TECHNIQUES = ["resnet50", "clip_vit_b32", "fashionclip", "siglip_base"]
TECHNIQUE_LABELS = {
    "resnet50": "ResNet50",
    "clip_vit_b32": "CLIP ViT-B/32",
    "fashionclip": "FashionCLIP",
    "siglip_base": "SigLIP",
}
TIERS = ["head", "mid", "tail"]

# dataviz categorical palette (light mode).
COLOR_BASELINE = "#2a78d6"   # slot 1 blue
COLOR_RETRIEVED = "#eb6834"  # slot 2 orange
COLOR_GROUNDTRUTH = "#1baf7a"  # slot 3 aqua
TECH_COLORS = {
    "resnet50": "#2a78d6",     # slot 1 blue
    "clip_vit_b32": "#eb6834",  # slot 2 orange
    "fashionclip": "#1baf7a",   # slot 3 aqua
    "siglip_base": "#eda100",   # slot 4 yellow
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_lookup():
    df = pd.read_csv(LOOKUP_CSV)
    ref_count = dict(zip(df["asin"], df["ref_count"]))
    tier = dict(zip(df["asin"], df["tier"]))
    return ref_count, tier


def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    also_related = {}
    for _, row in df.iterrows():
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        also_viewed = ast.literal_eval(row["also_viewed"]) if pd.notna(row["also_viewed"]) else []
        also_related[row["asin"]] = list(set(also_buy) | set(also_viewed))
    return df, also_related


def tier_composition(asins, tier_lookup):
    counts = Counter()
    n_unknown = 0
    for a in asins:
        t = tier_lookup.get(a)
        if t is None:
            n_unknown += 1
            continue
        counts[t] += 1
    total = sum(counts.values())
    frac = {t: (counts[t] / total if total else 0.0) for t in TIERS}
    return frac, total, n_unknown


def main():
    ref_count, tier_lookup = load_lookup()
    sample_df, also_related = load_sample()
    sample_asins = sample_df["asin"].tolist()

    with open(RETRIEVALS_JSON) as f:
        retrievals = json.load(f)

    # --- Baseline composition (technique-independent) ---
    baseline_frac, baseline_n, baseline_unknown = tier_composition(sample_asins, tier_lookup)

    # --- Ground truth composition (technique-independent) ---
    gt_asins = [a for asins in also_related.values() for a in asins]
    gt_frac, gt_n, gt_unknown = tier_composition(gt_asins, tier_lookup)

    # --- Per-technique: retrieved composition, ARP, coverage ---
    per_technique = {}
    for technique in TECHNIQUES:
        if technique not in retrievals:
            print(f"WARNING: {technique} missing from full_retrievals.json, skipping")
            continue
        entries = retrievals[technique]

        top5_pool, top10_pool = [], []
        for e in entries:
            top5_pool.extend(e["retrieved_top10"][:5])
            top10_pool.extend(e["retrieved_top10"][:10])

        retrieved5_frac, retrieved5_n, retrieved5_unknown = tier_composition(top5_pool, tier_lookup)
        retrieved10_frac, retrieved10_n, retrieved10_unknown = tier_composition(top10_pool, tier_lookup)

        arp5 = float(np.mean([ref_count.get(a, 0) for a in top5_pool]))
        arp10 = float(np.mean([ref_count.get(a, 0) for a in top10_pool]))

        coverage5 = len(set(top5_pool)) / len(sample_asins)
        coverage10 = len(set(top10_pool)) / len(sample_asins)

        per_technique[technique] = {
            "retrieved5_frac": retrieved5_frac,
            "retrieved10_frac": retrieved10_frac,
            "retrieved5_unknown": retrieved5_unknown,
            "arp5": arp5,
            "arp10": arp10,
            "coverage5": coverage5,
            "coverage10": coverage10,
            "n_unique_retrieved5": len(set(top5_pool)),
            "n_unique_retrieved10": len(set(top10_pool)),
        }
        print(f"{technique}: ARP@5={arp5:.2f} ARP@10={arp10:.2f} "
              f"coverage@5={coverage5:.3f} coverage@10={coverage10:.3f}")

    write_results_table(per_technique)
    write_bucket_composition(baseline_frac, baseline_n, gt_frac, gt_n, per_technique)
    plot_bucket_composition_per_technique(baseline_frac, gt_frac, per_technique)
    plot_arp_coverage_comparison(per_technique)


def write_results_table(per_technique):
    lines = [
        "# Phase 3 Results: ARP and Catalog Coverage by Technique",
        "",
        "**Average Recommendation Popularity (ARP)**: mean catalog-wide reference "
        "count (from `popularity_lookup.csv`) of items appearing in recommendations, "
        "averaged across all 1,872 queries. Higher = systematically recommending more "
        "popular items.",
        "",
        "**Catalog coverage**: fraction of the 1,872-product sample that appears at "
        "least once, anywhere, across every query's top-K recommendations pooled "
        "together. Low coverage = a small subset of products dominates recommendations "
        "regardless of query -- the direct signature of the overspecialization/repetition "
        "problem this phase was asked to check for.",
        "",
        "| Technique | ARP@5 | ARP@10 | Unique products recommended @5 | Coverage@5 | "
        "Unique products recommended @10 | Coverage@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TECHNIQUES:
        if t not in per_technique:
            continue
        d = per_technique[t]
        lines.append(
            f"| {TECHNIQUE_LABELS[t]} | {d['arp5']:.2f} | {d['arp10']:.2f} | "
            f"{d['n_unique_retrieved5']} | {d['coverage5']:.3f} | "
            f"{d['n_unique_retrieved10']} | {d['coverage10']:.3f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_MD}")


def write_bucket_composition(baseline_frac, baseline_n, gt_frac, gt_n, per_technique):
    lines = [
        "# Phase 3: Head/Mid/Tail Composition -- Sample vs Retrieved vs Ground Truth",
        "",
        "Tiers are catalog-wide (from step 1+2's full metadata pass), not sample-relative.",
        "",
        "## Baseline: composition of the 1,872-product sample itself",
        "",
        "This is the reference point -- if retrieval were popularity-agnostic, retrieved "
        "items should roughly match this composition.",
        "",
        "| Tier | % of sample |",
        "|---|---|",
    ]
    for tier in TIERS:
        lines.append(f"| {tier} | {100 * baseline_frac[tier]:.1f}% |")
    lines.append(f"\n(n={baseline_n} sample products with a known tier)")

    lines.append("")
    lines.append("## Ground truth: composition of real also_buy/also_viewed targets")
    lines.append("")
    lines.append("Aggregated across every query's actual relatedness edges (not the retrieved "
                 "items) -- shows what the \"correct answers\" look like in terms of real "
                 "popularity, independent of any model.")
    lines.append("")
    lines.append("| Tier | % of ground truth targets |")
    lines.append("|---|---|")
    for tier in TIERS:
        lines.append(f"| {tier} | {100 * gt_frac[tier]:.1f}% |")
    lines.append(f"\n(n={gt_n} ground-truth also_buy/also_viewed references with a known tier)")

    for t in TECHNIQUES:
        if t not in per_technique:
            continue
        d = per_technique[t]
        lines.append("")
        lines.append(f"## {TECHNIQUE_LABELS[t]}: retrieved composition (top-5, top-10)")
        lines.append("")
        lines.append("| Tier | % of top-5 retrieved | % of top-10 retrieved |")
        lines.append("|---|---|---|")
        for tier in TIERS:
            lines.append(
                f"| {tier} | {100 * d['retrieved5_frac'][tier]:.1f}% | "
                f"{100 * d['retrieved10_frac'][tier]:.1f}% |"
            )

    BUCKET_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {BUCKET_MD}")


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_bucket_composition_per_technique(baseline_frac, gt_frac, per_technique):
    x = np.arange(len(TIERS))
    width = 0.26

    for t in TECHNIQUES:
        if t not in per_technique:
            continue
        retrieved_frac = per_technique[t]["retrieved5_frac"]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
        _style_axis(ax)

        baseline_vals = [100 * baseline_frac[tier] for tier in TIERS]
        retrieved_vals = [100 * retrieved_frac[tier] for tier in TIERS]
        gt_vals = [100 * gt_frac[tier] for tier in TIERS]

        b1 = ax.bar(x - width, baseline_vals, width, label="Sample baseline",
                     color=COLOR_BASELINE, edgecolor=SURFACE, linewidth=2)
        b2 = ax.bar(x, retrieved_vals, width, label="Retrieved (top-5)",
                     color=COLOR_RETRIEVED, edgecolor=SURFACE, linewidth=2)
        b3 = ax.bar(x + width, gt_vals, width, label="Ground truth",
                     color=COLOR_GROUNDTRUTH, edgecolor=SURFACE, linewidth=2)

        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                        ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in TIERS], color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel("% of items", color=INK_SECONDARY, fontsize=10)
        ax.set_ylim(0, max(baseline_vals + retrieved_vals + gt_vals) * 1.2 + 5)
        ax.set_title(f"{TECHNIQUE_LABELS[t]}: Baseline vs Retrieved vs Ground Truth",
                     color=INK_PRIMARY, fontsize=12.5, fontweight="bold", loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")

        fig.tight_layout()
        out_path = PLOTS_DIR / f"bucket_composition_{t}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_arp_coverage_comparison(per_technique):
    techniques = [t for t in TECHNIQUES if t in per_technique]
    x = np.arange(len(techniques))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

    # Panel 1: ARP@5 / ARP@10
    ax = axes[0]
    _style_axis(ax)
    arp5 = [per_technique[t]["arp5"] for t in techniques]
    arp10 = [per_technique[t]["arp10"] for t in techniques]
    for i, t in enumerate(techniques):
        color = TECH_COLORS[t]
        ax.bar(x[i] - width / 2, arp5[i], width, color=color, alpha=0.55, edgecolor=SURFACE, linewidth=1.5)
        ax.bar(x[i] + width / 2, arp10[i], width, color=color, alpha=1.0, edgecolor=SURFACE, linewidth=1.5)
        ax.text(x[i] - width / 2, arp5[i] + max(arp10) * 0.01, f"{arp5[i]:.1f}",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
        ax.text(x[i] + width / 2, arp10[i] + max(arp10) * 0.01, f"{arp10[i]:.1f}",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([TECHNIQUE_LABELS[t] for t in techniques], color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Average Recommendation Popularity (mean ref_count)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("ARP by Technique", color=INK_PRIMARY, fontsize=12.5, fontweight="bold", loc="left")
    ax.set_ylim(0, max(arp5 + arp10) * 1.18)

    # Panel 2: coverage@5 / coverage@10
    ax = axes[1]
    _style_axis(ax)
    cov5 = [100 * per_technique[t]["coverage5"] for t in techniques]
    cov10 = [100 * per_technique[t]["coverage10"] for t in techniques]
    for i, t in enumerate(techniques):
        color = TECH_COLORS[t]
        ax.bar(x[i] - width / 2, cov5[i], width, color=color, alpha=0.55, edgecolor=SURFACE, linewidth=1.5)
        ax.bar(x[i] + width / 2, cov10[i], width, color=color, alpha=1.0, edgecolor=SURFACE, linewidth=1.5)
        ax.text(x[i] - width / 2, cov5[i] + max(cov10) * 0.01, f"{cov5[i]:.0f}%",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
        ax.text(x[i] + width / 2, cov10[i] + max(cov10) * 0.01, f"{cov10[i]:.0f}%",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([TECHNIQUE_LABELS[t] for t in techniques], color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Catalog coverage (% of 1,872-product sample)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_title("Catalog Coverage by Technique", color=INK_PRIMARY, fontsize=12.5, fontweight="bold", loc="left")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor=INK_MUTED, alpha=0.55, label="Top-5"),
        Patch(facecolor=INK_MUTED, alpha=1.0, label="Top-10"),
    ], frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, loc="upper center",
        bbox_to_anchor=(0.5, 1.04), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = PLOTS_DIR / "arp_coverage_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

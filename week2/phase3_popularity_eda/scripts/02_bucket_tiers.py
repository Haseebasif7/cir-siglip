"""
Phase 3, Step 2: bucket the full catalog into head/mid/tail popularity tiers
and plot the long-tail distribution.

Tiering is rank-based (true quantile split over the full sorted catalog),
not a fixed count-value threshold: head = top 20% of products by rank when
sorted descending by ref_count, mid = next 30%, tail = bottom 50%. Ties at
the cut boundary (expected -- see notes) mean the reported count ranges for
adjacent tiers can overlap; that's reported as-is rather than forced to
split cleanly, per the phase brief.

Rewrites popularity_lookup.csv in place with an added `rank` and `tier`
column so any later step can look up a tier for an asin with a single
dict/CSV load, without redoing the quantile math.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOOKUP_CSV = BASE_DIR / "data" / "popularity_lookup.csv"
PLOTS_DIR = BASE_DIR / "plots"
BUCKET_MD = BASE_DIR / "data" / "catalog_tier_summary.md"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HEAD_FRAC = 0.20
MID_FRAC = 0.30  # tail is the remainder (0.50)

# dataviz categorical palette (light mode), fixed order -- reused for the
# tier dimension across every chart in this phase for a consistent identity.
COLOR_HEAD = "#2a78d6"   # slot 1 blue
COLOR_MID = "#eb6834"    # slot 2 orange
COLOR_TAIL = "#1baf7a"   # slot 3 aqua
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def assign_tiers(df):
    n = len(df)
    head_cut = round(n * HEAD_FRAC)
    mid_cut = round(n * (HEAD_FRAC + MID_FRAC))

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1  # 1 = most-referenced
    tiers = ["head"] * head_cut + ["mid"] * (mid_cut - head_cut) + ["tail"] * (n - mid_cut)
    df["tier"] = tiers
    return df, head_cut, mid_cut


def write_summary(df, head_cut, mid_cut):
    n = len(df)
    lines = [
        "# Phase 3, Step 2: Catalog-Wide Popularity Tiers",
        "",
        f"Full catalog: {n:,} unique asins (from `popularity_lookup.csv`, "
        "step 1's single streaming pass over the full metadata file).",
        "",
        "Tiers are assigned by **rank** (quantile split over the full sorted-descending "
        "list: head = top 20% by rank, mid = next 30%, tail = bottom 50%), not by a fixed "
        "count-value threshold. Ties at the cut boundary are common -- a huge share of the "
        "catalog has ref_count 0 or 1 -- so adjacent tiers' reported count ranges can overlap; "
        "this is reported honestly rather than forced into a clean split.",
        "",
        "| Tier | N products | % of catalog | Min ref_count | Max ref_count | Median ref_count |",
        "|---|---|---|---|---|---|",
    ]
    for tier in ["head", "mid", "tail"]:
        sub = df[df["tier"] == tier]
        lines.append(
            f"| {tier} | {len(sub):,} | {100 * len(sub) / n:.1f}% | "
            f"{sub['ref_count'].min()} | {sub['ref_count'].max()} | "
            f"{sub['ref_count'].median():.1f} |"
        )
    lines.append("")
    lines.append(f"Rank cutoffs: head/mid boundary at rank {head_cut:,}, mid/tail boundary at rank {mid_cut:,}.")

    n_zero = (df["ref_count"] == 0).sum()
    lines.append("")
    lines.append(f"**{n_zero:,} products ({100 * n_zero / n:.1f}% of the full catalog) are never "
                 "referenced by any other product's also_buy/also_viewed at all** (ref_count == 0). "
                 "This is the true shape of the long tail across the whole catalog, not just the "
                 "1,872-product sample.")

    BUCKET_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {BUCKET_MD}")


def plot_long_tail(df):
    sorted_counts = df["ref_count"].to_numpy()
    ranks = df["rank"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(ranks, sorted_counts + 1, linewidth=2, color=COLOR_HEAD, solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Product rank (log scale, 1 = most-referenced)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Reference count + 1 (log scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Catalog-Wide Long Tail: also_buy/also_viewed Reference Counts",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left")

    n = len(df)
    head_cut = round(n * HEAD_FRAC)
    mid_cut = round(n * (HEAD_FRAC + MID_FRAC))
    ymax = ax.get_ylim()[1]
    for cutoff, label, y_frac in [(head_cut, "head | mid boundary", 1.06), (mid_cut, "mid | tail boundary", 1.13)]:
        ax.axvline(cutoff, color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.6)
        ax.annotate(label, xy=(cutoff, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(cutoff, y_frac), textcoords=("data", "axes fraction"),
                    color=INK_MUTED, fontsize=8, ha="center", va="bottom", annotation_clip=False)

    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.8)
    ax.grid(True, which="minor", color=GRIDLINE, linewidth=0.4, alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    fig.text(0.01, 0.01,
              f"n = {n:,} unique asins across the full Clothing/Shoes/Jewelry catalog. "
              "y-axis offset by +1 to plot zero-reference products on a log scale.",
              color=INK_MUTED, fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = PLOTS_DIR / "long_tail_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = pd.read_csv(LOOKUP_CSV)
    df = df.sort_values("ref_count", ascending=False, kind="stable")
    df, head_cut, mid_cut = assign_tiers(df)

    df.to_csv(LOOKUP_CSV, index=False)
    print(f"Rewrote {LOOKUP_CSV} with rank + tier columns.")

    write_summary(df, head_cut, mid_cut)
    plot_long_tail(df)


if __name__ == "__main__":
    main()

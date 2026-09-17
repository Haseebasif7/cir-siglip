"""
Phase 11, step 5: qualitative check on the four specific example queries
already discussed across phases 8, 9, and 10 -- the Tommy Hilfiger watch
(B005NGRC0W), the pink drawstring laundry bag (B01FWDLMYC), the Batgirl
costume (B01B5BIU0E), and the kurta (B01AIT77RQ) -- under five
configurations: Raw SigLIP / Phase 8 alone / Phase 9 alone / the best
overall blend from step 4 (alpha=0.5 raw + Phase 8, chosen for having the
best cross-type Hit Rate@5 of any blend while tying raw SigLIP on full-view
Hit Rate@5 -- see results_table.md) / the diversity-forcing configuration.

Directly checks whether step 4's aggregate numbers match what these
previously-discussed cases actually look like.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"
BUNDLE_NPZ = BASE_DIR / "data" / "embeddings_bundle.npz"
OUT_DIR = BASE_DIR / "qualitative_examples"

TOP_K = 5
BEST_ALPHA = 0.5  # phase 8 blend, see results_table.md section 1/2

QUERY_ASINS = ["B005NGRC0W", "B01FWDLMYC", "B01B5BIU0E", "B01AIT77RQ"]


def topk_excluding_self(sims, query_idx, k):
    row = sims[query_idx].copy()
    row[query_idx] = -np.inf
    top = np.argsort(-row)[:k]
    return top, row


def build_diversity_forced_top5(asins, raw_sims, types, query_idx):
    """Same rule as 02_evaluate_all_configs.py's build_diversity_forced_lists,
    inlined here for a single query for the qualitative grid."""
    q = asins[query_idx]
    row = raw_sims[query_idx].copy()
    row[query_idx] = -np.inf
    order = np.argsort(-row)
    raw_ranked = [asins[j] for j in order]

    my_type = types.get(q)
    if my_type is None:
        return raw_ranked[:5], row, "unknown_own_type"

    top4 = raw_ranked[:4]
    top4_set = set(top4)
    diverse_item = None
    for cand in raw_ranked:
        if cand in top4_set:
            continue
        if types.get(cand) is not None and types.get(cand) != my_type:
            diverse_item = cand
            break

    if diverse_item is None:
        return raw_ranked[:5], row, "no_cross_type_candidate"

    return top4 + [diverse_item], row, "ok"


def build_grid(query_asin, query_img_path, query_title, rows_config, image_path_map, title_map, out_path):
    fig, axes = plt.subplots(len(rows_config), TOP_K + 1, figsize=(3 * (TOP_K + 1), 3.2 * len(rows_config)))
    if len(rows_config) == 1:
        axes = axes.reshape(1, -1)

    for row_i, (row_label, top_asins, sim_lookup) in enumerate(rows_config):
        ax = axes[row_i, 0]
        try:
            ax.imshow(Image.open(query_img_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
        ax.axis("off")
        ax.set_title(f"QUERY\n{query_title[:30]}", fontsize=7)
        ax.text(-0.3, 0.5, row_label, transform=ax.transAxes, fontsize=9, ha="right", va="center", weight="bold")

        for col_i, asin_j in enumerate(top_asins):
            ax2 = axes[row_i, col_i + 1]
            img_path_j = EVAL_DIR / image_path_map[asin_j]
            title_j = str(title_map.get(asin_j, ""))
            try:
                ax2.imshow(Image.open(img_path_j).convert("RGB"))
            except Exception:
                ax2.text(0.5, 0.5, "[missing]", ha="center", va="center")
            ax2.axis("off")
            sim_str = f"sim={sim_lookup[asin_j]:.3f}" if sim_lookup is not None else ""
            ax2.set_title(f"#{col_i+1} {sim_str}\n{title_j[:28]}", fontsize=6)

    fig.suptitle(f"Amazon query: {query_asin}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    import ast
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    title_map = dict(zip(df["asin"], df["title"]))
    image_path_map = dict(zip(df["asin"], df["image_path"]))
    types = {}
    for _, row in df.iterrows():
        cats = ast.literal_eval(row["category"])
        types[row["asin"]] = cats[3] if len(cats) > 3 else None

    bundle = np.load(BUNDLE_NPZ, allow_pickle=True)
    asins = [str(a) for a in bundle["asins"]]
    raw, p8, p9 = bundle["raw"], bundle["phase8"], bundle["phase9"]
    asin_to_idx = {a: i for i, a in enumerate(asins)}

    raw_sims = raw @ raw.T
    p8_sims = p8 @ p8.T
    p9_sims = p9 @ p9.T
    blend_sims = BEST_ALPHA * raw_sims + (1 - BEST_ALPHA) * p8_sims

    for query_asin in QUERY_ASINS:
        if query_asin not in asin_to_idx:
            print(f"WARNING: {query_asin} not in eval sample, skipping.")
            continue
        qi = asin_to_idx[query_asin]

        rows_config = []
        for label, sims in [
            ("Raw SigLIP\n(alone)", raw_sims),
            ("Phase 8 alone\n(Amazon-trained)", p8_sims),
            ("Phase 9 alone\n(Polyvore-trained)", p9_sims),
            (f"Best blend\n(alpha={BEST_ALPHA} raw+P8)", blend_sims),
        ]:
            top, row = topk_excluding_self(sims, qi, TOP_K)
            top_asins = [asins[j] for j in top]
            sim_lookup = {asins[j]: row[j] for j in top}
            rows_config.append((label, top_asins, sim_lookup))

        div_top5, div_row, div_status = build_diversity_forced_top5(asins, raw_sims, types, qi)
        sim_lookup = {a: div_row[asin_to_idx[a]] for a in div_top5}
        rows_config.append((f"Diversity-forcing\n({div_status})", div_top5, sim_lookup))

        query_title = str(title_map.get(query_asin, ""))
        out_path = OUT_DIR / f"amazon_query_{query_asin}.png"
        build_grid(query_asin, EVAL_DIR / image_path_map[query_asin], query_title,
                   rows_config, image_path_map, title_map, out_path)


if __name__ == "__main__":
    main()

"""
Phase 15, preprocessing (plan step 2 / 4b): CSA-Net's `embed_from_feature`
requires a (cat_s, cat_t) category pair -- there is no category-free
embedding mode. Phase 12c's `nn_lookup.npz` (top-50 raw-SigLIP neighbors per
item) is NOT category-restricted, but the CIR eval task always retrieves
within a single category pool, so unrestricted negatives here would repeat
phase 14's documented train/eval mismatch failure mode. This script filters
each item's top-50 neighbors down to same-category-only, and reports the
resulting neighbor-count distribution so a minimum-neighbor threshold can be
chosen BEFORE training (not discovered as a problem afterward).

Item->category lookup built the same way as phase 13b's 03_csa_cir_eval.py:
phase 13's training_data.json (train+val items_by_category) first, falling
back to phase 9's Polyvore item metadata for any item nn_lookup covers that
training_data.json doesn't.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12C_DIR = BASE_DIR.parent / "phase12c_ranking_distillation"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"

NN_LOOKUP_NPZ = PHASE12C_DIR / "data" / "nn_lookup.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"
METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"

OUT_NPZ = BASE_DIR / "data" / "same_category_neighbors.npz"
OUT_MD = BASE_DIR / "data" / "neighbor_preprocessing_report.md"

MIN_NEIGHBORS = 5  # decided threshold: anchors with fewer same-category neighbors than
                    # this are excluded from substitute-loss sampling (too few candidates
                    # for a meaningful K-way ranking-distillation target).
MAX_K = 50  # nn_lookup's own top-K


def build_item_cat_lookup(categories):
    with open(TRAINING_DATA_JSON) as f:
        td = json.load(f)
    lookup = {}
    for split_key in ("train_items_by_category", "val_items_by_category"):
        for cat, items in td[split_key].items():
            for i in items:
                lookup[i] = cat
    with open(METADATA_JSON) as f:
        meta = json.load(f)
    n_from_meta = 0
    for item_id, v in meta.items():
        cat = v.get("semantic_category")
        if item_id not in lookup and cat in categories:
            lookup[item_id] = cat
            n_from_meta += 1
    print(f"item_cat_lookup: {len(lookup)} items ({len(lookup) - n_from_meta} from training_data.json, "
          f"{n_from_meta} from phase 9 metadata fallback)")
    return lookup


def main():
    with open(TRAINING_DATA_JSON) as f:
        categories = json.load(f)["categories"]

    item_cat = build_item_cat_lookup(categories)

    nn = np.load(NN_LOOKUP_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in nn["item_ids"]]
    indices = nn["indices"]  # (N, 50) int32, global indices into item_ids
    sims = nn["sims"].astype(np.float32)  # (N, 50)
    N = len(item_ids)
    print(f"nn_lookup: {N} items, K={indices.shape[1]}")

    cat_id = np.full(N, -1, dtype=np.int32)
    for i, iid in enumerate(item_ids):
        c = item_cat.get(iid)
        if c is not None:
            cat_id[i] = categories.index(c)
    n_uncategorized = int((cat_id == -1).sum())
    print(f"Items with no category found: {n_uncategorized} / {N}")

    # For each item, filter its neighbor list to same-category-only, right-padded
    # with -1 (invalid) so the result stays a dense (N, MAX_K) array.
    filt_indices = np.full((N, MAX_K), -1, dtype=np.int32)
    filt_sims = np.zeros((N, MAX_K), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)

    neighbor_cats = cat_id[indices]  # (N, K) -- category of each neighbor
    own_cats = cat_id[:, None]  # (N, 1)
    same_cat_mask = (neighbor_cats == own_cats) & (own_cats != -1)  # (N, K) bool

    for i in range(N):
        m = same_cat_mask[i]
        n_same = int(m.sum())
        counts[i] = n_same
        if n_same > 0:
            filt_indices[i, :n_same] = indices[i][m]
            filt_sims[i, :n_same] = sims[i][m]

    valid = cat_id != -1
    counts_valid = counts[valid]
    below_threshold = int((counts_valid < MIN_NEIGHBORS).sum())
    pct_below = 100.0 * below_threshold / max(1, valid.sum())

    pctiles = np.percentile(counts_valid, [0, 5, 25, 50, 75, 95, 100])
    print(f"Same-category neighbor count distribution (n={valid.sum()} categorized items): "
          f"min={pctiles[0]:.0f} p5={pctiles[1]:.0f} p25={pctiles[2]:.0f} median={pctiles[3]:.0f} "
          f"p75={pctiles[4]:.0f} p95={pctiles[5]:.0f} max={pctiles[6]:.0f}")
    print(f"Items below MIN_NEIGHBORS={MIN_NEIGHBORS}: {below_threshold} ({pct_below:.2f}%)")

    per_cat_stats = []
    for ci, cname in enumerate(categories):
        cat_mask = valid & (cat_id == ci)
        n_cat = int(cat_mask.sum())
        if n_cat == 0:
            continue
        c_counts = counts[cat_mask]
        n_below = int((c_counts < MIN_NEIGHBORS).sum())
        per_cat_stats.append((cname, n_cat, float(np.median(c_counts)), n_below, 100.0 * n_below / n_cat))
        print(f"  {cname}: n={n_cat} median_same_cat_neighbors={np.median(c_counts):.0f} "
              f"below_threshold={n_below} ({100.0*n_below/n_cat:.2f}%)")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        item_ids=np.array(item_ids),
        indices=filt_indices,
        sims=filt_sims,
        counts=counts,
        cat_id=cat_id,
    )
    print(f"Saved {OUT_NPZ}")

    lines = [
        "# Phase 15: Same-Category Neighbor Preprocessing Report",
        "",
        "CSA-Net's `embed_from_feature` requires a (cat_s, cat_t) category pair -- there is "
        "no category-free embedding mode. Phase 12c's `nn_lookup.npz` top-50 raw-SigLIP "
        "neighbors are NOT category-restricted, so this filters each item's neighbor list "
        "down to same-category-only before it's used as a substitute-loss training target, "
        "deliberately avoiding phase 14's documented train/eval negative-distribution "
        "mismatch (unrestricted training negatives vs. a category-restricted eval task).",
        "",
        f"- Catalog size (nn_lookup): {N}",
        f"- Items with a resolved category: {int(valid.sum())} ({n_uncategorized} uncategorized, excluded)",
        f"- Original K (nn_lookup): {MAX_K}",
        "",
        "## Same-category neighbor count distribution (after filtering, categorized items only)",
        "",
        f"min={pctiles[0]:.0f}, p5={pctiles[1]:.0f}, p25={pctiles[2]:.0f}, median={pctiles[3]:.0f}, "
        f"p75={pctiles[4]:.0f}, p95={pctiles[5]:.0f}, max={pctiles[6]:.0f}",
        "",
        f"**MIN_NEIGHBORS threshold = {MIN_NEIGHBORS}** (decided before training, not tuned after "
        "seeing downstream results): anchors with fewer than this many same-category neighbors "
        "are excluded from substitute-loss sampling during training, rather than injecting a "
        "noisy, near-degenerate K-way ranking target.",
        "",
        f"- Items below threshold: {below_threshold} / {int(valid.sum())} ({pct_below:.2f}%)",
        "",
        "## Per-category breakdown",
        "",
        "| Category | N items | Median same-cat neighbors | Below threshold | % below |",
        "|---|---|---|---|---|",
    ]
    for cname, n_cat, med, n_below, pct in per_cat_stats:
        lines.append(f"| {cname} | {n_cat} | {med:.0f} | {n_below} | {pct:.2f}% |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

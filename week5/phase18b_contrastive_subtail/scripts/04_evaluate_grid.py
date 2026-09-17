"""Phase 18b, steps 4-5: evaluate the trained model. Same 5x5 grid,
methodology, gallery, and queries as phase 18's `05_evaluate_grid.py` --
directly comparable numbers, not a different protocol.

Step 4 (this phase's own specific, narrow check, done FIRST and highlighted
separately before anything broader): does sub_tail's PURE corner
(alpha1=1.0, alpha2=0.0) now retrieve lower mean ref_count / higher tail
fraction than sub_rel's PURE corner (alpha1=1.0, alpha2=1.0)? Phase 18 found
this reversed even at the unblended corner level (sub_tail ref_count=17.23 >
sub_rel's 16.86) -- this is the specific, narrow thing this phase needs to
fix, checked directly before evaluating anything else.

Step 5: the same corner-sanity / axis-independence / smoothness checks phase
18 ran, to see whether the fix also resolves the broader axis-2 reversal.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CORNER_ALPHAS, FourHeadDedicatedCapacity

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE16_DIR = BASE_DIR.parent / "phase16_relevance_tail_dial"
PHASE1B_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"

CHECKPOINT = BASE_DIR / "models" / "contrastive_subtail.pt"
GALLERY_NPZ = PHASE16_DIR / "data" / "candidate_gallery.npz"
QUERY_NPZ = PHASE1B_DIR / "embeddings" / "siglip_base.npz"
QUERY_CSV = PHASE1B_DIR / "data" / "sample_data.csv"

OUT_CACHE = BASE_DIR / "data" / "grid_topk_cache.npz"
CORNER_MD = BASE_DIR / "corner_sanity_checks.md"
AXIS_MD = BASE_DIR / "axis_independence_check.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
K_CACHE = 50
K_EVAL = 10

# Phase 18's own corner numbers, for direct before/after comparison
PHASE18_CORNERS = {
    "sub_rel": {"hit_rate": 0.4594, "mean_ref_count": 16.86, "tail_fraction": 0.3415, "overlap_with_raw": 0.6674},
    "sub_tail": {"hit_rate": 0.4546, "mean_ref_count": 17.23, "tail_fraction": 0.3303, "overlap_with_raw": 0.6638},
    "comp_rel": {"hit_rate": 0.4311, "mean_ref_count": 18.84, "tail_fraction": 0.3417, "overlap_with_raw": 0.4343},
    "comp_tail": {"hit_rate": 0.3381, "mean_ref_count": 15.39, "tail_fraction": 0.3675, "overlap_with_raw": 0.3047},
}


def load_relatedness():
    df = pd.read_csv(QUERY_CSV)
    related = {}
    for _, row in df.iterrows():
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        also_viewed = ast.literal_eval(row["also_viewed"]) if pd.notna(row["also_viewed"]) else []
        related[row["asin"]] = set(also_buy) | set(also_viewed)
    return related


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    model = FourHeadDedicatedCapacity().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    g = np.load(GALLERY_NPZ, allow_pickle=True)
    gallery_asins = g["asins"].astype(str)
    gallery_emb_raw = l2norm(g["embeddings"])

    q = np.load(QUERY_NPZ, allow_pickle=True)
    query_asins = q["asins"].astype(str)
    query_emb_raw = l2norm(q["embeddings"])

    gallery_idx = {a: i for i, a in enumerate(gallery_asins)}
    self_rows, self_cols = [], []
    for qi, qa in enumerate(query_asins):
        if qa in gallery_idx:
            self_rows.append(qi)
            self_cols.append(gallery_idx[qa])
    print(f"Gallery: {len(gallery_asins)} items. Queries: {len(query_asins)}. Self-matches: {len(self_rows)}")

    related = load_relatedness()
    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count", "tier"])
    tier_df["asin"] = tier_df["asin"].astype(str)
    refcount_lookup = dict(zip(tier_df["asin"], tier_df["ref_count"].astype(int)))
    tail_lookup = dict(zip(tier_df["asin"], tier_df["tier"] == "tail"))

    gallery_t = torch.tensor(gallery_emb_raw, device=DEVICE)
    query_t = torch.tensor(query_emb_raw, device=DEVICE)

    with torch.no_grad():
        raw_sims = query_t @ gallery_t.T
        if self_rows:
            raw_sims[self_rows, self_cols] = -float("inf")
        _, raw_topk_idx = torch.topk(raw_sims, K_EVAL, dim=1)
    raw_topk_asins = gallery_asins[raw_topk_idx.cpu().numpy()]

    all_topk_asins = np.empty((len(GRID), len(GRID), len(query_asins), K_CACHE), dtype=gallery_asins.dtype)
    grid_metrics = {}

    for i1, a1 in enumerate(GRID):
        for i2, a2 in enumerate(GRID):
            with torch.no_grad():
                z_gallery = model(gallery_t, a1, a2)
                z_query = model(query_t, a1, a2)
                sims = z_query @ z_gallery.T
                if self_rows:
                    sims[self_rows, self_cols] = -float("inf")
                topk_sims, topk_idx = torch.topk(sims, K_CACHE, dim=1)
            topk_idx_np = topk_idx.cpu().numpy()
            retrieved = gallery_asins[topk_idx_np]
            all_topk_asins[i1, i2] = retrieved

            top10 = retrieved[:, :K_EVAL]
            hits, refcounts, tail_flags, overlap_raw_vals = [], [], [], []
            for qi, qa in enumerate(query_asins):
                items = top10[qi]
                gt = related.get(qa, set())
                hits.append(1 if any(it in gt for it in items) else 0)
                refcounts.extend(refcount_lookup.get(it, 0) for it in items)
                tail_flags.extend(1 if tail_lookup.get(it, False) else 0 for it in items)
                overlap_raw_vals.append(len(set(items) & set(raw_topk_asins[qi])) / K_EVAL)

            m = {
                "hit_rate": float(np.mean(hits)),
                "mean_ref_count": float(np.mean(refcounts)),
                "tail_fraction": float(np.mean(tail_flags)),
                "overlap_with_raw": float(np.mean(overlap_raw_vals)),
            }
            grid_metrics[(a1, a2)] = m
            print(f"alpha1={a1:.2f} alpha2={a2:.2f}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    np.savez(OUT_CACHE, grid=np.array(GRID), query_asins=query_asins, topk_asins=all_topk_asins)
    print(f"Cached grid retrievals: {OUT_CACHE}")

    corner_vals = {corner: grid_metrics[alphas] for corner, alphas in CORNER_ALPHAS.items()}
    sr, st, cr, ct = corner_vals["sub_rel"], corner_vals["sub_tail"], corner_vals["comp_rel"], corner_vals["comp_tail"]

    # ---------- Step 4: the pure-corner check, highlighted first ----------
    step4_fixed = st["mean_ref_count"] < sr["mean_ref_count"] and st["tail_fraction"] > sr["tail_fraction"]
    corner_lines = [
        "# Phase 18b, Step 4 & Step 6: Corner Sanity Checks\n",
        "## Step 4 (this phase's specific, narrow target -- checked FIRST, before anything broader)\n",
        "Does sub_tail's PURE corner now retrieve lower mean ref_count / higher tail fraction than "
        "sub_rel's PURE corner? Phase 18 found this reversed even unblended (sub_tail ref_count=17.23 > "
        "sub_rel's 16.86).\n",
        "| | sub_rel (alpha1=1.0, alpha2=1.0) | sub_tail (alpha1=1.0, alpha2=0.0) |",
        "|---|---|---|",
        f"| Mean ref_count@10 -- phase 18 | {PHASE18_CORNERS['sub_rel']['mean_ref_count']:.2f} | "
        f"{PHASE18_CORNERS['sub_tail']['mean_ref_count']:.2f} |",
        f"| Mean ref_count@10 -- **phase 18b** | {sr['mean_ref_count']:.2f} | **{st['mean_ref_count']:.2f}** |",
        f"| Tail fraction@10 -- phase 18 | {PHASE18_CORNERS['sub_rel']['tail_fraction']:.4f} | "
        f"{PHASE18_CORNERS['sub_tail']['tail_fraction']:.4f} |",
        f"| Tail fraction@10 -- **phase 18b** | {sr['tail_fraction']:.4f} | **{st['tail_fraction']:.4f}** |",
        "",
        f"**{'FIXED' if step4_fixed else 'STILL REVERSED'}**: sub_tail's ref_count "
        f"({st['mean_ref_count']:.2f}) is now {'lower' if st['mean_ref_count'] < sr['mean_ref_count'] else 'NOT lower'} "
        f"than sub_rel's ({sr['mean_ref_count']:.2f}), and sub_tail's tail_fraction ({st['tail_fraction']:.4f}) is "
        f"{'higher' if st['tail_fraction'] > sr['tail_fraction'] else 'NOT higher'} than sub_rel's "
        f"({sr['tail_fraction']:.4f}).\n",
        "## Step 6: full corner sanity table (same format as phase 18)\n",
        "| Corner | (alpha1, alpha2) | Hit Rate@10 | Mean ref_count@10 | Tail fraction@10 | "
        "Overlap w/ raw SigLIP@10 |",
        "|---|---|---|---|---|---|",
    ]
    for corner, (a1, a2) in CORNER_ALPHAS.items():
        m = corner_vals[corner]
        corner_lines.append(f"| {corner} | ({a1}, {a2}) | {m['hit_rate']:.4f} | {m['mean_ref_count']:.2f} | "
                             f"{m['tail_fraction']:.4f} | {m['overlap_with_raw']:.4f} |")
    corner_lines.append("")
    corner_lines.append("## Direct before/after comparison to phase 18\n")
    corner_lines.append("| Corner | Metric | Phase 18 | Phase 18b |")
    corner_lines.append("|---|---|---|---|")
    for corner in ["sub_rel", "sub_tail", "comp_rel", "comp_tail"]:
        for metric in ["hit_rate", "mean_ref_count", "tail_fraction", "overlap_with_raw"]:
            fmt = ".2f" if metric == "mean_ref_count" else ".4f"
            corner_lines.append(f"| {corner} | {metric} | {format(PHASE18_CORNERS[corner][metric], fmt)} | "
                                 f"{format(corner_vals[corner][metric], fmt)} |")
    corner_lines.append("")
    CORNER_MD.write_text("\n".join(corner_lines))
    print(f"Saved {CORNER_MD}")
    print(f"Step 4 verdict: {'FIXED' if step4_fixed else 'STILL REVERSED'}")

    # ---------- Step 5: axis independence, same format as phase 18 ----------
    axis_lines = ["# Phase 18b, Step 5: Axis Independence Check (Re-Run)\n",
                  "Same method as phase 18's step 7. For each fixed alpha2, does moving alpha1 alone reproduce "
                  "the same axis-1 behavior regardless of alpha2? And the reverse for axis 2.\n"]

    axis_lines.append("## Axis 1 (substitute <-> complement) at each fixed alpha2\n")
    axis_lines.append("| alpha2 | overlap_raw(a1=0) | overlap_raw(a1=1) | delta | hit_rate(a1=0) | hit_rate(a1=1) | delta |")
    axis_lines.append("|---|---|---|---|---|---|---|")
    axis1_overlap_deltas, axis1_hitrate_deltas = [], []
    for a2 in GRID:
        m0, m1 = grid_metrics[(0.0, a2)], grid_metrics[(1.0, a2)]
        d_overlap = m1["overlap_with_raw"] - m0["overlap_with_raw"]
        d_hit = m0["hit_rate"] - m1["hit_rate"]
        axis1_overlap_deltas.append(d_overlap)
        axis1_hitrate_deltas.append(d_hit)
        axis_lines.append(f"| {a2} | {m0['overlap_with_raw']:.4f} | {m1['overlap_with_raw']:.4f} | {d_overlap:+.4f} | "
                           f"{m0['hit_rate']:.4f} | {m1['hit_rate']:.4f} | {d_hit:+.4f} |")
    axis_lines.append("")
    overlap_consistent = all(d > 0 for d in axis1_overlap_deltas)
    hit_consistent = all(d > 0 for d in axis1_hitrate_deltas)
    axis_lines.append(f"**Axis 1 direction consistent across every alpha2 value: overlap-with-raw "
                       f"{'YES' if overlap_consistent else 'NO'} (deltas: {[round(d,4) for d in axis1_overlap_deltas]}), "
                       f"hit-rate {'YES' if hit_consistent else 'NO'} (deltas: {[round(d,4) for d in axis1_hitrate_deltas]}). "
                       f"Delta range: overlap [{min(axis1_overlap_deltas):.4f}, {max(axis1_overlap_deltas):.4f}], "
                       f"hit-rate [{min(axis1_hitrate_deltas):.4f}, {max(axis1_hitrate_deltas):.4f}].**\n")

    axis_lines.append("## Axis 2 (relevance <-> tail-exposure) at each fixed alpha1\n")
    axis_lines.append("| alpha1 | ref_count(a2=0) | ref_count(a2=1) | delta | tail_frac(a2=0) | tail_frac(a2=1) | delta |")
    axis_lines.append("|---|---|---|---|---|---|---|")
    axis2_refcount_deltas, axis2_tailfrac_deltas = [], []
    for a1 in GRID:
        m0, m1 = grid_metrics[(a1, 0.0)], grid_metrics[(a1, 1.0)]
        d_ref = m1["mean_ref_count"] - m0["mean_ref_count"]
        d_tail = m0["tail_fraction"] - m1["tail_fraction"]
        axis2_refcount_deltas.append(d_ref)
        axis2_tailfrac_deltas.append(d_tail)
        axis_lines.append(f"| {a1} | {m0['mean_ref_count']:.2f} | {m1['mean_ref_count']:.2f} | {d_ref:+.2f} | "
                           f"{m0['tail_fraction']:.4f} | {m1['tail_fraction']:.4f} | {d_tail:+.4f} |")
    axis_lines.append("")
    ref_consistent = all(d > 0 for d in axis2_refcount_deltas)
    tail_consistent = all(d > 0 for d in axis2_tailfrac_deltas)
    axis_lines.append(f"**Axis 2 direction consistent across every alpha1 value: ref_count "
                       f"{'YES' if ref_consistent else 'NO'} (deltas: {[round(d,2) for d in axis2_refcount_deltas]}), "
                       f"tail_fraction {'YES' if tail_consistent else 'NO'} (deltas: {[round(d,4) for d in axis2_tailfrac_deltas]}). "
                       f"Delta range: ref_count [{min(axis2_refcount_deltas):.2f}, {max(axis2_refcount_deltas):.2f}], "
                       f"tail_fraction [{min(axis2_tailfrac_deltas):.4f}, {max(axis2_tailfrac_deltas):.4f}].**\n")

    all_independent = overlap_consistent and hit_consistent and ref_consistent and tail_consistent
    axis_lines.append("## Verdict\n")
    if all_independent:
        axis_lines.append("**The two axes behave independently.** Every directional check holds in the correct "
                           "direction at EVERY value of the other axis.")
    else:
        axis_lines.append("**The two axes do NOT behave fully independently** -- at least one directional check "
                           "failed at some value of the other axis. Reported plainly.")
    axis_lines.append("")
    AXIS_MD.write_text("\n".join(axis_lines))
    print(f"Saved {AXIS_MD}")
    print(f"Axis independence verdict: {all_independent}")


if __name__ == "__main__":
    main()

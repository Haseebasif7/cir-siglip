"""Phase 18, steps 6-7: evaluate the trained model across a 5x5 grid of
(alpha1, alpha2) -- coarser than the project's usual 11-point 1D sweeps,
per the brief's own suggestion for the larger 2D space. Computes, per grid
point, four diagnostics over phase 1b's 1,872-item query sample against
phase 16's 26,591-item merged candidate gallery (both reused directly, not
rebuilt):

- `overlap_with_raw`: mean top-10 overlap between the model's retrieval and
  raw (unprojected) SigLIP's own top-10 retrieval -- the substitute-axis
  "visual similarity" signal (phase 12d/17's own diagnostic, adapted).
- `hit_rate`: fraction of queries whose top-10 retrieval contains a real
  also_buy/also_viewed ground-truth item -- the complement-axis
  "co-occurrence" signal (phase 16/16d's own diagnostic).
- `mean_ref_count`: mean catalog-wide ref_count (phase 3) of the top-10
  retrieved items -- the tail-exposure-axis signal (phase 16/16d's own).
- `tail_fraction`: fraction of the top-10 retrieved items that are
  tail-tier by phase 3's definition -- a second, directly interpretable
  tail-exposure-axis signal.

Retrievals are cached (top-50 per query per grid point) to
`data/grid_topk_cache.npz` for the smoothness-2D script (06) to reuse
without re-running the model.

Writes `corner_sanity_checks.md` (step 6: the 4 exact grid corners only,
checked against what each corner's parent mechanism should look like) and
`axis_independence_check.md` (step 7: does moving one alpha alone produce
consistent axis behavior regardless of the other alpha's value -- the
central question this phase exists to answer).
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

CHECKPOINT = BASE_DIR / "models" / "unified_two_axis.pt"
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

# Reference numbers from the parent mechanisms, for step 6's "does this corner resemble its parent" check.
# Phase 16d (relevance/tail dial, Amazon, dedicated capacity): Hit Rate@10 relevance=0.3948, tail=0.3029;
# mean_ref_count@5 relevance=18.04, tail=15.86 (see phase16d results_table.md).
PHASE16D_REFERENCE = {"hit_rate_relevance": 0.3948, "hit_rate_tail": 0.3029,
                       "ref_count_relevance": 18.04, "ref_count_tail": 15.86}
# Phase 17 (substitute/complement dial, Polyvore, dedicated capacity): overlap_with_raw substitute=0.3310,
# complement=0.1132 (see phase17 results_table.md step 5.2) -- different dataset, cited as directional
# context only (substitute >> complement on visual-similarity-to-raw), not a matched-protocol number.
PHASE17_REFERENCE_DIRECTION = {"overlap_raw_substitute_gt_complement": True}


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

    # Raw SigLIP top-K per query (alpha-independent, computed once)
    raw_gallery_t = gallery_t
    raw_query_t = query_t
    with torch.no_grad():
        raw_sims = raw_query_t @ raw_gallery_t.T
        if self_rows:
            raw_sims[self_rows, self_cols] = -float("inf")
        _, raw_topk_idx = torch.topk(raw_sims, K_EVAL, dim=1)
    raw_topk_asins = gallery_asins[raw_topk_idx.cpu().numpy()]  # (n_queries, K_EVAL)

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

    # ---------- Step 6: corner sanity checks ----------
    corner_lines = ["# Phase 18, Step 6: Corner Sanity Checks\n",
                     "Each corner checked directly against what its parent mechanism(s) should look like -- "
                     "not assumed to work just because the axes work individually.\n",
                     "| Corner | (alpha1, alpha2) | Hit Rate@10 | Mean ref_count@10 | Tail fraction@10 | "
                     "Overlap w/ raw SigLIP@10 |",
                     "|---|---|---|---|---|---|"]
    corner_vals = {}
    for corner, (a1, a2) in CORNER_ALPHAS.items():
        m = grid_metrics[(a1, a2)]
        corner_vals[corner] = m
        corner_lines.append(f"| {corner} | ({a1}, {a2}) | {m['hit_rate']:.4f} | {m['mean_ref_count']:.2f} | "
                             f"{m['tail_fraction']:.4f} | {m['overlap_with_raw']:.4f} |")
    corner_lines.append("")

    sr, st, cr, ct = corner_vals["sub_rel"], corner_vals["sub_tail"], corner_vals["comp_rel"], corner_vals["comp_tail"]

    corner_lines.append("## sub_rel -- should resemble phase 17's substitute mode (high visual similarity to raw SigLIP)\n")
    corner_lines.append(f"Overlap w/ raw SigLIP = {sr['overlap_with_raw']:.4f}. Phase 17's Polyvore substitute mode "
                         f"measured 0.3310 (different dataset, cited as directional context only). "
                         f"**{'Consistent' if sr['overlap_with_raw'] > cr['overlap_with_raw'] else 'INCONSISTENT'}**: "
                         f"sub_rel's overlap-with-raw ({sr['overlap_with_raw']:.4f}) is "
                         f"{'higher' if sr['overlap_with_raw'] > cr['overlap_with_raw'] else 'NOT higher'} than "
                         f"comp_rel's ({cr['overlap_with_raw']:.4f}), as a substitute-mode corner should be.\n")

    corner_lines.append("## comp_rel -- should resemble phase 16/16d's relevance mode (real co-occurrence signal)\n")
    corner_lines.append(f"Hit Rate@10 = {cr['hit_rate']:.4f}. Phase 16d's relevance-mode Hit Rate@10 was "
                         f"{PHASE16D_REFERENCE['hit_rate_relevance']:.4f} (same also_buy signal, 2-head "
                         "architecture, direct numeric comparison valid since this is the same underlying "
                         f"signal and gallery/query setup). "
                         f"**{'Consistent' if cr['hit_rate'] > st['hit_rate'] and cr['hit_rate'] > sr['hit_rate'] else 'Check directly'}**: "
                         f"comp_rel's Hit Rate ({cr['hit_rate']:.4f}) vs sub_rel's ({sr['hit_rate']:.4f}) and "
                         f"sub_tail's ({st['hit_rate']:.4f}).\n")

    corner_lines.append("## comp_tail -- should resemble phase 16d's tail mode (lower ref_count, real co-occurrence)\n")
    corner_lines.append(f"Mean ref_count@10 = {ct['mean_ref_count']:.2f}, Hit Rate@10 = {ct['hit_rate']:.4f}. "
                         f"Phase 16d's tail-mode reference: ref_count@5={PHASE16D_REFERENCE['ref_count_tail']:.2f}, "
                         f"Hit Rate@10={PHASE16D_REFERENCE['hit_rate_tail']:.4f} (same attribute-pair signal, "
                         "2-head architecture). "
                         f"**{'Consistent' if ct['mean_ref_count'] < cr['mean_ref_count'] else 'INCONSISTENT'}**: "
                         f"comp_tail's ref_count ({ct['mean_ref_count']:.2f}) is "
                         f"{'lower' if ct['mean_ref_count'] < cr['mean_ref_count'] else 'NOT lower'} than "
                         f"comp_rel's ({cr['mean_ref_count']:.2f}).\n")

    corner_lines.append("## sub_tail -- THE NEW CORNER: should be visually similar AND skew toward low ref_count -- "
                         "checked directly, not assumed from its two parent behaviors\n")
    both_hold = st['overlap_with_raw'] > cr['overlap_with_raw'] and st['mean_ref_count'] < cr['mean_ref_count']
    corner_lines.append(f"Overlap w/ raw SigLIP = {st['overlap_with_raw']:.4f} (vs comp_rel's {cr['overlap_with_raw']:.4f}), "
                         f"mean ref_count@10 = {st['mean_ref_count']:.2f} (vs comp_rel's {cr['mean_ref_count']:.2f}), "
                         f"tail_fraction@10 = {st['tail_fraction']:.4f} (vs comp_rel's {cr['tail_fraction']:.4f}). "
                         f"**{'BOTH properties hold' if both_hold else 'AT LEAST ONE PROPERTY DOES NOT HOLD'}**: "
                         f"sub_tail retrieves items that are "
                         f"{'more' if st['overlap_with_raw'] > cr['overlap_with_raw'] else 'NOT more'} visually similar "
                         f"to the query AND "
                         f"{'lower' if st['mean_ref_count'] < cr['mean_ref_count'] else 'NOT lower'} in ref_count than "
                         "the plain complement-relevance corner -- this is the one genuinely new corner and its "
                         "behavior is reported exactly as measured, not assumed from sub_rel + comp_tail alone.\n")

    CORNER_MD.write_text("\n".join(corner_lines))
    print(f"Saved {CORNER_MD}")

    # ---------- Step 7: axis independence ----------
    axis_lines = ["# Phase 18, Step 7: Axis Independence Check -- The Central Question\n",
                  "For each fixed alpha2, does moving alpha1 alone reproduce the same axis-1 behavior "
                  "(overlap-with-raw increasing, hit rate decreasing) regardless of alpha2's value? And the "
                  "reverse: for each fixed alpha1, does moving alpha2 alone reproduce the same axis-2 behavior "
                  "(ref_count decreasing, tail_fraction increasing) regardless of alpha1's value?\n"]

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
        axis_lines.append("**The two axes behave independently.** Every directional check (axis 1's "
                           "overlap-with-raw and hit-rate; axis 2's ref_count and tail_fraction) holds in the "
                           "correct direction at EVERY value of the other axis, not just at its own endpoints. "
                           "Moving one alpha does not reverse or erase the other axis's effect.")
    else:
        axis_lines.append("**The two axes do NOT behave fully independently** -- at least one directional check "
                           "failed at some value of the other axis. Reported plainly: this means the two "
                           "dedicated-capacity axes interact to some degree even though each one works on its "
                           "own at its own endpoints.")
    axis_lines.append("")
    AXIS_MD.write_text("\n".join(axis_lines))
    print(f"Saved {AXIS_MD}")
    print(f"Axis independence verdict: {all_independent}")


if __name__ == "__main__":
    main()

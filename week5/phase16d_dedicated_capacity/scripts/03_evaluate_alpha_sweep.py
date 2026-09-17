"""11-point alpha sweep, identical methodology to phases 16/16c (same
gallery -- reused directly from phase 16's folder, model-independent raw
SigLIP embeddings -- same queries, same K values, same axis checks), for
direct three-way comparability.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DedicatedCapacityHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE16_DIR = BASE_DIR.parent / "phase16_relevance_tail_dial"
PHASE1B_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"

CHECKPOINT = BASE_DIR / "models" / "dedicated_capacity.pt"
GALLERY_NPZ = PHASE16_DIR / "data" / "candidate_gallery.npz"
QUERY_NPZ = PHASE1B_DIR / "embeddings" / "siglip_base.npz"
QUERY_CSV = PHASE1B_DIR / "data" / "sample_data.csv"

OUT_CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_RESULTS_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ALPHAS = [round(x, 1) for x in np.arange(0.0, 1.0001, 0.1)]
K_CACHE = 50
K_VALUES = [5, 10]

# phase 16 and 16c's endpoint numbers, for direct three-way comparison
PHASE16_REFERENCE = {
    0.0: {"hit_rate@5": 0.3194, "hit_rate@10": 0.3958, "mean_ref_count@5": 16.8983, "mean_ref_count@10": 16.5813},
    1.0: {"hit_rate@5": 0.3184, "hit_rate@10": 0.3948, "mean_ref_count@5": 17.4908, "mean_ref_count@10": 17.2126},
}
PHASE16C_REFERENCE = {
    0.0: {"hit_rate@5": 0.2810, "hit_rate@10": 0.3563, "mean_ref_count@5": 17.9713, "mean_ref_count@10": 17.3860},
    1.0: {"hit_rate@5": 0.2783, "hit_rate@10": 0.3510, "mean_ref_count@5": 17.5932, "mean_ref_count@10": 17.2651},
}
# phase 16's own best_val_rel (validation relevance loss at convergence), the quality-non-degradation reference
PHASE16_BEST_VAL_REL = 0.9735


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
    model = DedicatedCapacityHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    g = np.load(GALLERY_NPZ, allow_pickle=True)
    gallery_asins = g["asins"].astype(str)
    gallery_emb = l2norm(g["embeddings"])

    q = np.load(QUERY_NPZ, allow_pickle=True)
    query_asins = q["asins"].astype(str)
    query_emb = l2norm(q["embeddings"])

    gallery_idx = {a: i for i, a in enumerate(gallery_asins)}
    self_match_rows, self_match_cols = [], []
    for qi, qa in enumerate(query_asins):
        if qa in gallery_idx:
            self_match_rows.append(qi)
            self_match_cols.append(gallery_idx[qa])
    print(f"Gallery: {len(gallery_asins)} items (reused from phase 16). Queries: {len(query_asins)}. "
          f"Self-matches to exclude: {len(self_match_rows)}")

    related = load_relatedness()
    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count"])
    refcount_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["ref_count"].astype(int)))

    gallery_t = torch.tensor(gallery_emb, device=DEVICE)
    query_t = torch.tensor(query_emb, device=DEVICE)

    all_retrieved_asins = np.empty((len(ALPHAS), len(query_asins), K_CACHE), dtype=gallery_asins.dtype)
    all_retrieved_sims = np.empty((len(ALPHAS), len(query_asins), K_CACHE), dtype=np.float32)

    accuracy_rows = []
    for ai, alpha in enumerate(ALPHAS):
        with torch.no_grad():
            z_gallery = model(gallery_t, alpha=alpha)
            z_query = model(query_t, alpha=alpha)
            sims = z_query @ z_gallery.T
            if self_match_rows:
                sims[self_match_rows, self_match_cols] = -float("inf")
            topk_sims, topk_idx = torch.topk(sims, K_CACHE, dim=1)

        topk_idx_np = topk_idx.cpu().numpy()
        topk_sims_np = topk_sims.cpu().numpy()
        retrieved_asins = gallery_asins[topk_idx_np]
        all_retrieved_asins[ai] = retrieved_asins
        all_retrieved_sims[ai] = topk_sims_np

        row = {"alpha": alpha}
        for K in K_VALUES:
            hits = 0
            refcounts = []
            for qi, qa in enumerate(query_asins):
                topk_items = retrieved_asins[qi, :K]
                gt = related.get(qa, set())
                if any(item in gt for item in topk_items):
                    hits += 1
                refcounts.extend(refcount_lookup.get(item, 0) for item in topk_items)
            row[f"hit_rate@{K}"] = hits / len(query_asins)
            row[f"mean_ref_count@{K}"] = float(np.mean(refcounts))
        accuracy_rows.append(row)
        print(f"alpha={alpha:.1f}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "alpha"))

    np.savez(OUT_CACHE, alphas=np.array(ALPHAS), query_asins=query_asins,
             retrieved_asins=all_retrieved_asins, retrieved_sims=all_retrieved_sims)
    print(f"Cached retrievals: {OUT_CACHE}")

    row_rel = next(r for r in accuracy_rows if r["alpha"] == 1.0)
    row_tail = next(r for r in accuracy_rows if r["alpha"] == 0.0)
    axis1_pass = all(row_rel[f"hit_rate@{K}"] > row_tail[f"hit_rate@{K}"] for K in K_VALUES)
    axis2_pass = all(row_tail[f"mean_ref_count@{K}"] < row_rel[f"mean_ref_count@{K}"] for K in K_VALUES)
    axis1_gap_5 = row_rel["hit_rate@5"] - row_tail["hit_rate@5"]
    axis1_gap_10 = row_rel["hit_rate@10"] - row_tail["hit_rate@10"]
    axis1_clear = axis1_gap_5 > 0.02 and axis1_gap_10 > 0.02

    lines = ["# Phase 16d: Results Table\n", "## Retrieval accuracy and axis checks across the alpha sweep\n"]
    header = "| alpha | " + " | ".join(f"Hit Rate@{K}" for K in K_VALUES) + " | " + \
              " | ".join(f"Mean ref_count@{K}" for K in K_VALUES) + " |"
    sep = "|" + "---|" * (1 + 2 * len(K_VALUES))
    lines.append(header)
    lines.append(sep)
    for row in accuracy_rows:
        cells = [f"{row['alpha']:.1f}"] + [f"{row[f'hit_rate@{K}']:.4f}" for K in K_VALUES] + \
                [f"{row[f'mean_ref_count@{K}']:.2f}" for K in K_VALUES]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Direct three-way comparison to phases 16 and 16c\n")
    lines.append("| | Phase 16 tail | Phase 16 relevance | Phase 16c tail | Phase 16c relevance | "
                 "Phase 16d tail | Phase 16d relevance |")
    lines.append("|---|---|---|---|---|---|---|")
    for K in K_VALUES:
        lines.append(f"| Hit Rate@{K} | {PHASE16_REFERENCE[0.0][f'hit_rate@{K}']:.4f} | "
                     f"{PHASE16_REFERENCE[1.0][f'hit_rate@{K}']:.4f} | "
                     f"{PHASE16C_REFERENCE[0.0][f'hit_rate@{K}']:.4f} | "
                     f"{PHASE16C_REFERENCE[1.0][f'hit_rate@{K}']:.4f} | "
                     f"{row_tail[f'hit_rate@{K}']:.4f} | {row_rel[f'hit_rate@{K}']:.4f} |")
    for K in K_VALUES:
        lines.append(f"| Mean ref_count@{K} | {PHASE16_REFERENCE[0.0][f'mean_ref_count@{K}']:.2f} | "
                     f"{PHASE16_REFERENCE[1.0][f'mean_ref_count@{K}']:.2f} | "
                     f"{PHASE16C_REFERENCE[0.0][f'mean_ref_count@{K}']:.2f} | "
                     f"{PHASE16C_REFERENCE[1.0][f'mean_ref_count@{K}']:.2f} | "
                     f"{row_tail[f'mean_ref_count@{K}']:.2f} | {row_rel[f'mean_ref_count@{K}']:.2f} |")
    lines.append("")

    lines.append("## Axis checks (endpoints, alpha=1.0 relevance vs alpha=0.0 tail-exposure)\n")
    lines.append(f"- **Axis 1 (relevance mode's Hit Rate@K should exceed tail mode's, every K): "
                 f"{'PASS' if axis1_pass else 'FAIL'}** -- "
                 + ", ".join(f"K={K}: relevance={row_rel[f'hit_rate@{K}']:.4f} vs "
                              f"tail={row_tail[f'hit_rate@{K}']:.4f}" for K in K_VALUES))
    lines.append(f"- **Axis 1, CLEAR-GAP bar (same >0.02 threshold phase 16c used): "
                 f"{'CLEAR PASS' if axis1_clear else 'NOT CLEAR'}** -- gap@5={axis1_gap_5:.4f}, "
                 f"gap@10={axis1_gap_10:.4f}")
    lines.append(f"- **Axis 2 (tail mode's mean retrieved ref_count should be lower than relevance mode's, every "
                 f"K): {'PASS' if axis2_pass else 'FAIL'}** -- "
                 + ", ".join(f"K={K}: relevance={row_rel[f'mean_ref_count@{K}']:.2f} vs "
                              f"tail={row_tail[f'mean_ref_count@{K}']:.2f}" for K in K_VALUES))
    lines.append("")

    OUT_RESULTS_MD.write_text("\n".join(lines))
    print(f"Results written: {OUT_RESULTS_MD}")
    print(f"Axis 1: {'PASS' if axis1_pass else 'FAIL'} (clear: {axis1_clear}), Axis 2: {'PASS' if axis2_pass else 'FAIL'}")


if __name__ == "__main__":
    main()

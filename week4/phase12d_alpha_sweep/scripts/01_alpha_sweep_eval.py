"""
Phase 12d, step 1: sweep alpha at 11 points (0.0 to 1.0, step 0.1) using
phase 12c's already-trained checkpoint -- no retraining, evaluation only.

For every alpha, computes:
  1. Full CIR benchmark Recall@10/30/50 (all 29,681 queries, same harness as
     phases 12/12b/12c).
  2. The four step-1 diagnostic metrics, on FIXED query samples (same seeds
     as phases 12/12b/12c's own diagnostics) so differences across alpha
     reflect alpha itself, not sample variation:
     a. mean visual similarity to the query (top-10 avg raw SigLIP cosine),
        1,000-query sample.
     b. hit rate against real outfit co-occurrence ground truth, same
        1,000-query sample.
     c. top-10 overlap with raw SigLIP's own retrieval, 500-query sample.
     d. top-10 overlap with pure complement-mode (alpha=0.0) retrieval, same
        500-query sample.

Also caches each alpha's actual top-10 retrieved list per query (on the
500-query overlap sample) to `data/topk_cache.json`, reused by step 3's
adjacent-vs-distant overlap check so that script doesn't need to recompute
retrieval from scratch.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark, topk_for_query
from model import ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12C_DIR = BASE_DIR.parent / "phase12c_ranking_distillation"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT = PHASE12C_DIR / "models" / "ranking_distillation.pt"

RESULTS_MD = BASE_DIR / "results_table.md"
RESULTS_JSON = BASE_DIR / "data" / "alpha_sweep_results.json"
TOPK_CACHE_JSON = BASE_DIR / "data" / "topk_cache.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KS = (10, 30, 50)
TOP_K = 10
DIAG_SAMPLE_SIZE = 1000
OVERLAP_SAMPLE_SIZE = 500
SEED = 42
ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def load_raw_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


@torch.no_grad()
def project(model, embeddings, alpha, batch_size=4096):
    outs = []
    for start in range(0, len(embeddings), batch_size):
        x = torch.tensor(embeddings[start:start + batch_size], device=DEVICE)
        outs.append(model(x, alpha=alpha).cpu().numpy())
    return np.concatenate(outs, axis=0)


def query_raw_vec(raw_emb, idx, query_items):
    item_idx = [idx[i] for i in query_items if i in idx]
    if not item_idx:
        return None
    v = raw_emb[item_idx].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def main():
    pools, queries = load_benchmark()
    item_ids, raw_emb = load_raw_embeddings()
    idx = {a: i for i, a in enumerate(item_ids)}

    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    diag_rng = np.random.default_rng(SEED)
    diag_sample = diag_rng.choice(len(queries), size=min(DIAG_SAMPLE_SIZE, len(queries)), replace=False)
    overlap_rng = np.random.default_rng(SEED)
    overlap_sample = overlap_rng.choice(len(queries), size=OVERLAP_SAMPLE_SIZE, replace=False)

    # Fixed reference: pure complement-mode (alpha=0.0) retrieval on the overlap sample.
    comp_emb_ref = project(model, raw_emb, 0.0)
    comp_top_ref = {}
    for qi in overlap_sample:
        q = queries[qi]
        top, _ = topk_for_query(pools, q, item_ids, comp_emb_ref, k=TOP_K)
        comp_top_ref[int(qi)] = top

    all_results = {}
    topk_cache = {}  # alpha -> {qi: top10_ids}

    for alpha in ALPHAS:
        emb = project(model, raw_emb, alpha)

        recall, n, n_skip = evaluate_recall(pools, queries, item_ids, emb, ks=KS)

        # Diagnostic metrics a & b on the 1,000-query sample
        avg_visual_sims, hit_flags = [], []
        for qi in diag_sample:
            q = queries[qi]
            qrv = query_raw_vec(raw_emb, idx, q["query_items"])
            if qrv is None:
                continue
            top, _ = topk_for_query(pools, q, item_ids, emb, k=TOP_K)
            if not top:
                continue
            avg_visual_sims.append(float(np.mean(raw_emb[[idx[i] for i in top]] @ qrv)))
            hit_flags.append(int(q["target_item"] in top))
        axis1 = float(np.mean(avg_visual_sims))
        axis2 = float(np.mean(hit_flags))

        # Diagnostic metrics c & d, plus topk cache, on the 500-query sample
        overlap_raw_vals, overlap_comp_vals = [], []
        alpha_top_cache = {}
        for qi in overlap_sample:
            q = queries[qi]
            top, _ = topk_for_query(pools, q, item_ids, emb, k=TOP_K)
            raw_top, _ = topk_for_query(pools, q, item_ids, raw_emb, k=TOP_K)
            if not top or not raw_top:
                continue
            alpha_top_cache[int(qi)] = top
            overlap_raw_vals.append(len(set(top) & set(raw_top)) / TOP_K)
            overlap_comp_vals.append(len(set(top) & set(comp_top_ref[int(qi)])) / TOP_K)

        overlap_with_raw = float(np.mean(overlap_raw_vals))
        overlap_with_comp = float(np.mean(overlap_comp_vals))

        topk_cache[str(alpha)] = alpha_top_cache
        all_results[str(alpha)] = {
            "recall": recall, "n_queries": n, "n_skipped": n_skip,
            "axis1_visual_sim": axis1, "axis2_hit_rate": axis2,
            "overlap_with_raw": overlap_with_raw, "overlap_with_complement": overlap_with_comp,
        }
        print(f"alpha={alpha:.1f}: recall@10={recall[10]:.4f} axis1={axis1:.4f} axis2={axis2:.4f} "
              f"overlap_raw={overlap_with_raw:.4f} overlap_comp={overlap_with_comp:.4f}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {RESULTS_JSON}")

    with open(TOPK_CACHE_JSON, "w") as f:
        json.dump(topk_cache, f)
    print(f"Saved {TOPK_CACHE_JSON}")

    lines = [
        "# Phase 12d: Alpha Interpolation Sweep -- Results",
        "",
        f"Phase 12c's `ranking_distillation.pt` checkpoint, evaluated at 11 alpha values "
        f"(0.0 to 1.0, step 0.1) -- no retraining. Full CIR benchmark: all "
        f"{sum(len(v) for v in pools.values())} pool slots, {len(queries)} queries. "
        f"Diagnostic metrics a/b on the same {DIAG_SAMPLE_SIZE}-query sample (seed={SEED}) "
        f"used throughout phases 12/12b/12c; metrics c/d on the same "
        f"{OVERLAP_SAMPLE_SIZE}-query overlap sample (seed={SEED}).",
        "",
        "| alpha | Recall@10 | Recall@30 | Recall@50 | Visual sim (a) | Co-occur hit rate (b) | Overlap w/ raw SigLIP (c) | Overlap w/ complement (d) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for alpha in ALPHAS:
        r = all_results[str(alpha)]
        rec = r["recall"]  # still int-keyed in memory, pre-JSON-round-trip
        lines.append(
            f"| {alpha:.1f} | {rec[10]:.4f} | {rec[30]:.4f} | {rec[50]:.4f} "
            f"| {r['axis1_visual_sim']:.4f} | {r['axis2_hit_rate']:.4f} "
            f"| {r['overlap_with_raw']:.4f} | {r['overlap_with_complement']:.4f} |"
        )
    lines.append("")
    lines.append(
        "Expected directions per the brief: (a) increase as alpha rises toward 1; "
        "(b) increase as alpha falls toward 0; (c) increase as alpha rises toward 1; "
        "(d) decrease as alpha rises toward 1 (trivially 1.0 at alpha=0.0 itself, since "
        "that IS the complement-mode reference). See `monotonicity_check.md` for whether "
        "each metric actually holds this direction consistently across the full sweep, "
        "not just at the two endpoints."
    )
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_MD}")


if __name__ == "__main__":
    main()

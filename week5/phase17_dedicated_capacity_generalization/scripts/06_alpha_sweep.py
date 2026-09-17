"""
Phase 17, step 5.2: 11-point alpha sweep (0.0-1.0, step 0.1) on this phase's
dedicated-capacity checkpoint, phase 12d's exact method
(week4/phase12d_alpha_sweep/scripts/01_alpha_sweep_eval.py) -- full CIR
benchmark Recall@K plus the four diagnostic metrics (visual similarity to
query, co-occurrence hit rate, overlap with raw SigLIP, overlap with pure
complement mode) at every alpha, on the same fixed 1000/500-query samples
(seed=42) phases 12/12b/12c/12d used, so differences reflect alpha itself,
not sampling noise. Also caches each alpha's top-10 retrieval per query for
`07_smoothness_check.py`.

Appended to `results_table.md` (run after 03_evaluate_cir.py and
04_official_polyvore_benchmark.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark, topk_for_query
from model import DedicatedCapacityHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT = BASE_DIR / "models" / "dedicated_capacity_substitute_complement.pt"

RESULTS_MD = BASE_DIR / "results_table.md"
RESULTS_JSON = BASE_DIR / "data" / "alpha_sweep_results.json"
TOPK_CACHE_JSON = BASE_DIR / "data" / "topk_cache.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KS = (10, 30, 50)
TOP_K = 10
DIAG_SAMPLE_SIZE = 1000
OVERLAP_SAMPLE_SIZE = 500
SEED = 42
ALPHAS = [round(0.1 * i, 1) for i in range(11)]

# Phase 12d's own sweep endpoints (same checkpoint's alpha=0.0/1.0 rows, for direct comparison)
PHASE12D_ENDPOINTS = {
    0.0: {"recall10": 0.0971, "axis1": 0.7217, "axis2": 0.1020, "overlap_raw": 0.1708, "overlap_comp": 1.0000},
    1.0: {"recall10": 0.0667, "axis1": 0.7498, "axis2": 0.0760, "overlap_raw": 0.3428, "overlap_comp": 0.4022},
}


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

    model = DedicatedCapacityHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    diag_rng = np.random.default_rng(SEED)
    diag_sample = diag_rng.choice(len(queries), size=min(DIAG_SAMPLE_SIZE, len(queries)), replace=False)
    overlap_rng = np.random.default_rng(SEED)
    overlap_sample = overlap_rng.choice(len(queries), size=OVERLAP_SAMPLE_SIZE, replace=False)

    comp_emb_ref = project(model, raw_emb, 0.0)
    comp_top_ref = {}
    for qi in overlap_sample:
        q = queries[qi]
        top, _ = topk_for_query(pools, q, item_ids, comp_emb_ref, k=TOP_K)
        comp_top_ref[int(qi)] = top

    all_results = {}
    topk_cache = {}

    for alpha in ALPHAS:
        emb = project(model, raw_emb, alpha)
        recall, n, n_skip = evaluate_recall(pools, queries, item_ids, emb, ks=KS)

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
              f"overlap_raw={overlap_with_raw:.4f} overlap_comp={overlap_with_comp:.4f}", flush=True)

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {RESULTS_JSON}")

    with open(TOPK_CACHE_JSON, "w") as f:
        json.dump(topk_cache, f)
    print(f"Saved {TOPK_CACHE_JSON}")

    lines = [
        "",
        "---",
        "",
        "## Step 5.2: Alpha Interpolation Sweep",
        "",
        f"This phase's `dedicated_capacity_substitute_complement.pt` checkpoint, evaluated at 11 "
        f"alpha values (0.0 to 1.0, step 0.1) -- no retraining. Full CIR benchmark: all "
        f"{sum(len(v) for v in pools.values())} pool slots, {len(queries)} queries. Diagnostic "
        f"metrics a/b on the same {DIAG_SAMPLE_SIZE}-query sample (seed={SEED}) used throughout "
        f"phases 12/12b/12c/12d; metrics c/d on the same {OVERLAP_SAMPLE_SIZE}-query overlap "
        f"sample (seed={SEED}).",
        "",
        "| alpha | Recall@10 | Recall@30 | Recall@50 | Visual sim (a) | Co-occur hit rate (b) | Overlap w/ raw SigLIP (c) | Overlap w/ complement (d) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for alpha in ALPHAS:
        r = all_results[str(alpha)]
        rec = r["recall"]
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
        "that IS the complement-mode reference)."
    )
    lines.append("")
    lines.append("### Endpoint comparison against phase 12d's own sweep (same checkpoint family, shared-trunk architecture)")
    lines.append("")
    lines.append("| Metric | Phase 12d alpha=0.0 (complement) | Phase 17 alpha=0.0 (complement) | Phase 12d alpha=1.0 (substitute) | Phase 17 alpha=1.0 (substitute) |")
    lines.append("|---|---|---|---|---|")
    r0, r1 = all_results["0.0"], all_results["1.0"]
    lines.append(f"| Recall@10 | {PHASE12D_ENDPOINTS[0.0]['recall10']:.4f} | {r0['recall'][10]:.4f} | "
                  f"{PHASE12D_ENDPOINTS[1.0]['recall10']:.4f} | {r1['recall'][10]:.4f} |")
    lines.append(f"| Visual sim (a) | {PHASE12D_ENDPOINTS[0.0]['axis1']:.4f} | {r0['axis1_visual_sim']:.4f} | "
                  f"{PHASE12D_ENDPOINTS[1.0]['axis1']:.4f} | {r1['axis1_visual_sim']:.4f} |")
    lines.append(f"| Co-occur hit rate (b) | {PHASE12D_ENDPOINTS[0.0]['axis2']:.4f} | {r0['axis2_hit_rate']:.4f} | "
                  f"{PHASE12D_ENDPOINTS[1.0]['axis2']:.4f} | {r1['axis2_hit_rate']:.4f} |")
    lines.append(f"| Overlap w/ raw SigLIP (c) | {PHASE12D_ENDPOINTS[0.0]['overlap_raw']:.4f} | {r0['overlap_with_raw']:.4f} | "
                  f"{PHASE12D_ENDPOINTS[1.0]['overlap_raw']:.4f} | {r1['overlap_with_raw']:.4f} |")
    axis1_gap_12d = PHASE12D_ENDPOINTS[1.0]['axis1'] - PHASE12D_ENDPOINTS[0.0]['axis1']
    axis1_gap_17 = r1['axis1_visual_sim'] - r0['axis1_visual_sim']
    axis2_gap_12d = PHASE12D_ENDPOINTS[0.0]['axis2'] - PHASE12D_ENDPOINTS[1.0]['axis2']
    axis2_gap_17 = r0['axis2_hit_rate'] - r1['axis2_hit_rate']
    decisive_gap_12d = PHASE12D_ENDPOINTS[1.0]['overlap_raw'] - PHASE12D_ENDPOINTS[0.0]['overlap_raw']
    decisive_gap_17 = r1['overlap_with_raw'] - r0['overlap_with_raw']
    lines.append("")
    lines.append(f"- Axis 1 gap (substitute - complement, visual sim): phase 12d = {axis1_gap_12d:.4f}, "
                  f"**phase 17 = {axis1_gap_17:.4f}**")
    lines.append(f"- Axis 2 gap (complement - substitute, hit rate): phase 12d = {axis2_gap_12d:.4f}, "
                  f"**phase 17 = {axis2_gap_17:.4f}**")
    lines.append(f"- Decisive gap (substitute-vs-raw minus complement-vs-raw overlap): phase 12d = "
                  f"{decisive_gap_12d:.4f}, **phase 17 = {decisive_gap_17:.4f}** "
                  f"(phase 12c's own stopping-condition threshold for a 'real' gap was 0.03)")
    lines.append("")

    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended to {RESULTS_MD}")


if __name__ == "__main__":
    main()

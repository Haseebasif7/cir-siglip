"""
Phase 21, step 5 (data collection half): 11-point alpha sweep (0.0-1.0,
step 0.1), blending the TWO FULLY INDEPENDENT heads' outputs at inference
time -- no shared trunk to condition, so blending happens purely as a
post-hoc combination of two already-computed embeddings:

    z_blend = normalize((1 - alpha) * z_complement + alpha * z_substitute)

alpha=0.0 reproduces the complement head exactly (normalize(z_complement) ==
z_complement, already unit norm); alpha=1.0 reproduces the substitute head
exactly -- same endpoint-identity property every additive-mode-vector dial
in this project has had, just applied to two independently-parameterized
networks' final outputs instead of one shared trunk plus a learned
correction. This is the genuinely open architectural question this phase
exists to answer: does this still interpolate coherently, or does it
collapse into a hard switch between two unrelated embedding spaces (which
would just be a version of phase 11's blending, already identified as not
a real architectural contribution)?

Same method and diagnostics as phase 12d/17's own sweeps (full CIR
benchmark Recall@K, the two axis diagnostics, overlap with raw SigLIP,
overlap with pure complement mode, on the same fixed samples, seed=42),
plus caching each alpha's top-10 retrieval per query for
`05_smoothness_check.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark, topk_for_query
from model import ProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
COMPLEMENT_CKPT = BASE_DIR / "models" / "complement_head.pt"
SUBSTITUTE_CKPT = BASE_DIR / "models" / "substitute_head.pt"

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
BATCH = 4096

PHASE9_RECALL10 = 0.1317
PHASE17_ENDPOINTS = {
    0.0: {"recall10": 0.1202, "axis1": 0.7142, "axis2": 0.1260, "overlap_raw": 0.1132},
    1.0: {"recall10": 0.0573, "axis1": 0.7492, "axis2": 0.0710, "overlap_raw": 0.3310},
}


def load_raw_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


@torch.no_grad()
def project_all(model, raw_emb):
    outs = []
    for start in range(0, len(raw_emb), BATCH):
        x = torch.tensor(raw_emb[start:start + BATCH], device=DEVICE)
        outs.append(model(x).cpu().numpy())
    return np.concatenate(outs, axis=0)


def blend(z_comp, z_sub, alpha):
    combined = (1 - alpha) * z_comp + alpha * z_sub
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (combined / norms).astype(np.float32)


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

    comp_model = ProjectionHead().to(DEVICE).eval()
    comp_model.load_state_dict(torch.load(COMPLEMENT_CKPT, map_location=DEVICE))
    sub_model = ProjectionHead().to(DEVICE).eval()
    sub_model.load_state_dict(torch.load(SUBSTITUTE_CKPT, map_location=DEVICE))

    z_comp_all = project_all(comp_model, raw_emb)
    z_sub_all = project_all(sub_model, raw_emb)
    print(f"Projected {len(item_ids)} items through both independent heads.")

    diag_rng = np.random.default_rng(SEED)
    diag_sample = diag_rng.choice(len(queries), size=min(DIAG_SAMPLE_SIZE, len(queries)), replace=False)
    overlap_rng = np.random.default_rng(SEED)
    overlap_sample = overlap_rng.choice(len(queries), size=OVERLAP_SAMPLE_SIZE, replace=False)

    comp_emb_ref = blend(z_comp_all, z_sub_all, 0.0)
    comp_top_ref = {}
    for qi in overlap_sample:
        q = queries[qi]
        top, _ = topk_for_query(pools, q, item_ids, comp_emb_ref, k=TOP_K)
        comp_top_ref[int(qi)] = top

    all_results = {}
    topk_cache = {}

    for alpha in ALPHAS:
        emb = blend(z_comp_all, z_sub_all, alpha)
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
        print(f"alpha={alpha:.1f}: recall@10={recall[10]:.4f} recall@30={recall[30]:.4f} recall@50={recall[50]:.4f} "
              f"axis1={axis1:.4f} axis2={axis2:.4f} overlap_raw={overlap_with_raw:.4f} "
              f"overlap_comp={overlap_with_comp:.4f}", flush=True)

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {RESULTS_JSON}")

    with open(TOPK_CACHE_JSON, "w") as f:
        json.dump(topk_cache, f)
    print(f"Saved {TOPK_CACHE_JSON}")

    lines = [
        "# Phase 21: Alpha Sweep -- Blending Two Fully Independent Networks",
        "",
        f"Full CIR benchmark ({sum(len(v) for v in pools.values())} pool slots, {len(queries)} "
        f"queries), 11-point alpha sweep, `z_blend = normalize((1-alpha)*z_complement + "
        f"alpha*z_substitute)` -- the complement and substitute heads share ZERO parameters, "
        f"trained fully independently, this blend is the only place their outputs ever "
        f"interact. Diagnostic metrics a/b on the same {DIAG_SAMPLE_SIZE}-query sample "
        f"(seed={SEED}) used throughout phases 12/12b/12c/12d/17; metrics c/d on the same "
        f"{OVERLAP_SAMPLE_SIZE}-query overlap sample (seed={SEED}).",
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
        "Expected directions per this project's standing convention: (a) increase as alpha "
        "rises toward 1; (b) increase as alpha falls toward 0; (c) increase as alpha rises "
        "toward 1; (d) decrease as alpha rises toward 1 (trivially 1.0 at alpha=0.0 itself)."
    )
    lines.append("")
    lines.append("## Endpoints against phase 9 (complement, alone) and phase 17 (this project's best prior dial)")
    lines.append("")
    lines.append("| Configuration | Recall@10 | Recall@30 | Recall@50 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Phase 9 (standalone, no dial) | {PHASE9_RECALL10:.4f} | -- | -- |")
    lines.append(f"| Phase 17 complement endpoint (alpha=0.0, dedicated capacity, one shared layer) | "
                  f"{PHASE17_ENDPOINTS[0.0]['recall10']:.4f} | -- | -- |")
    r0, r1 = all_results["0.0"], all_results["1.0"]
    lines.append(f"| **Phase 21 complement endpoint (alpha=0.0, zero shared parameters)** | "
                  f"**{r0['recall'][10]:.4f}** | **{r0['recall'][30]:.4f}** | **{r0['recall'][50]:.4f}** |")
    lines.append(f"| Phase 17 substitute endpoint (alpha=1.0) | {PHASE17_ENDPOINTS[1.0]['recall10']:.4f} | -- | -- |")
    lines.append(f"| Phase 21 substitute endpoint (alpha=1.0, zero shared parameters) | "
                  f"{r1['recall'][10]:.4f} | {r1['recall'][30]:.4f} | {r1['recall'][50]:.4f} |")
    lines.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_MD}")


if __name__ == "__main__":
    main()

"""
Phase 15, step 6: alpha-sweep CIR evaluation for a trained
`CSANetSigLIPControllable` checkpoint. Combines phase 13b's per-context-item
average-pairwise-distance scoring (`03_csa_cir_eval.py`'s
`evaluate_csa_recall`, needed because CSA-Net's embeddings are category-pair-
conditioned, not a flat lookup) with phase 12d's alpha-sweep diagnostic suite
(`phase12d_alpha_sweep/scripts/01_alpha_sweep_eval.py`): visual-similarity
axis, co-occurrence hit-rate axis, overlap-with-raw-SigLIP, overlap-with-
alpha=0 reference, all on the same fixed diagnostic query samples (seed=42)
phase 12d used, for direct comparability.

Candidate embeddings are computed once per (category, alpha) via
`all_as_candidate_embeddings_from_feature` and reused for every query in that
category -- this is the expensive part of the sweep (see
`sweep_cost_estimate.md`, produced by timing a single alpha point first).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CSANetSigLIPControllable

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12_DIR = BASE_DIR.parent / "phase12_controllable_modes"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"

BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"
METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"

DEVICE = "cpu"  # measured faster than MPS for this workload: many small per-item ops
# dominate cost here, and MPS per-op dispatch overhead outweighs its raw compute
# advantage at this scale (measured: cpu 0.41s/step vs mps 1.35s/step, see training_log.md)
KS = (10, 30, 50)
TOP_K_DIAG = 10
DIAG_SAMPLE_SIZE = 1000
OVERLAP_SAMPLE_SIZE = 500
SEED = 42
ALPHAS = [round(0.1 * i, 1) for i in range(11)]


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
    for item_id, v in meta.items():
        cat = v.get("semantic_category")
        if item_id not in lookup and cat in categories:
            lookup[item_id] = cat
    return lookup


def load_features():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, embeddings


def evaluate_alpha(model, alpha, pools, queries, item_ids, base_features, raw_embeddings,
                    categories, item_cat_lookup, idx, diag_qidxs, overlap_qidxs,
                    alpha0_topk_ref=None):
    """One full pass over the benchmark at a fixed alpha. Returns
    (recall, n_total, n_skipped, diag_topk) where diag_topk maps qi -> list of
    top-K item ids (only populated for qi in diag_qidxs | overlap_qidxs, to
    avoid keeping the full benchmark's top-K in memory)."""
    cat_to_i = {c: i for i, c in enumerate(categories)}
    hits = {k: 0 for k in KS}
    n_total, n_skipped = 0, 0
    diag_topk = {}
    need_topk = set(diag_qidxs) | set(overlap_qidxs)

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    eye = torch.eye(len(categories), device=DEVICE)

    with torch.no_grad():
        for cat, qidxs in by_cat.items():
            if cat not in pools:
                n_skipped += len(qidxs)
                continue
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_gidx = [idx[i] for i in pool_ids if i in idx]
            if len(pool_gidx) != len(pool_ids):
                n_skipped += len(qidxs)
                continue
            x_pool = torch.tensor(base_features[pool_gidx], device=DEVICE)
            cat_t_fixed = torch.zeros(len(pool_gidx), len(categories), device=DEVICE)
            cat_t_fixed[:, cat_to_i[cat]] = 1.0
            cand_all = model.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed, alpha)

            for qi in qidxs:
                q = queries[qi]
                ctx_items = [i for i in q["query_items"] if i in idx]
                if not ctx_items or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                ctx_gidx = [idx[i] for i in ctx_items]
                x_ctx = torch.tensor(base_features[ctx_gidx], device=DEVICE)

                dist_sum = torch.zeros(len(pool_gidx), device=DEVICE)
                for ci, item_id in enumerate(ctx_items):
                    c_i = cat_to_i[item_cat_lookup[item_id]]
                    f_ctx = model.embed_from_feature(
                        x_ctx[ci:ci + 1], eye[c_i:c_i + 1], cat_t_fixed[0:1], alpha,
                    )
                    cand_slice = cand_all[:, c_i, :]
                    d = ((f_ctx - cand_slice) ** 2).sum(dim=-1)
                    dist_sum += d

                dist_avg = (dist_sum / len(ctx_items)).cpu().numpy()
                target_pos = pool_pos[q["target_item"]]
                target_dist = dist_avg[target_pos]
                rank = int((dist_avg <= target_dist).sum())
                n_total += 1
                for k in KS:
                    if rank <= k:
                        hits[k] += 1

                if qi in need_topk:
                    top_local = np.argsort(dist_avg)[:TOP_K_DIAG]
                    diag_topk[qi] = [pool_ids[i] for i in top_local]

    recall = {k: hits[k] / n_total for k in KS} if n_total else {k: 0.0 for k in KS}
    return recall, n_total, n_skipped, diag_topk


def raw_siglip_topk(pools, queries, item_ids, raw_embeddings, idx, qidxs, k=TOP_K_DIAG):
    """Raw-SigLIP-cosine top-K per query, for the overlap-with-raw diagnostic
    (alpha-independent, computed once)."""
    out = {}
    for qi in qidxs:
        q = queries[qi]
        cat = q["category"]
        pool_ids = pools.get(cat, [])
        pool_gidx = [idx[i] for i in pool_ids if i in idx]
        if len(pool_gidx) != len(pool_ids):
            continue
        ctx_gidx = [idx[i] for i in q["query_items"] if i in idx]
        if not ctx_gidx:
            continue
        qv = raw_embeddings[ctx_gidx].mean(axis=0)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        pool_vecs = raw_embeddings[pool_gidx]
        sims = pool_vecs @ qv
        top_local = np.argsort(-sims)[:k]
        out[qi] = [pool_ids[i] for i in top_local]
    return out


def diagnostics(queries, raw_embeddings, idx, qidxs, topk_by_qi):
    """Axis 1 (visual similarity) and axis 2 (co-occurrence hit rate), phase
    12d's definitions: axis1 = mean raw-SigLIP cosine of retrieved items to
    the query's own raw representation; axis2 = fraction of queries whose
    true target appears in the top-K."""
    sims, hits = [], []
    for qi in qidxs:
        if qi not in topk_by_qi:
            continue
        q = queries[qi]
        ctx_gidx = [idx[i] for i in q["query_items"] if i in idx]
        if not ctx_gidx:
            continue
        qv = raw_embeddings[ctx_gidx].mean(axis=0)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        top_ids = topk_by_qi[qi]
        top_gidx = [idx[i] for i in top_ids if i in idx]
        if top_gidx:
            sims.append(float(np.mean(raw_embeddings[top_gidx] @ qv)))
        hits.append(int(q["target_item"] in top_ids))
    return float(np.mean(sims)) if sims else float("nan"), float(np.mean(hits)) if hits else float("nan")


def overlap(a_topk, b_topk, qidxs, k=TOP_K_DIAG):
    vals = []
    for qi in qidxs:
        if qi in a_topk and qi in b_topk:
            vals.append(len(set(a_topk[qi]) & set(b_topk[qi])) / k)
    return float(np.mean(vals)) if vals else float("nan")


def run_sweep(checkpoint_path, label, out_json):
    with open(TRAINING_DATA_JSON) as f:
        categories = json.load(f)["categories"]
    item_cat_lookup = build_item_cat_lookup(categories)

    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    item_ids, raw_embeddings = load_features()
    idx = {a: i for i, a in enumerate(item_ids)}
    print(f"[{label}] Loaded {len(item_ids)} item features, {len(queries)} queries.")

    model = CSANetSigLIPControllable().to(DEVICE).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # encode_feature (the Linear(768,64) projection) has no alpha-dependence -- only
    # attn_net does -- so it's computed ONCE per checkpoint and reused across the whole
    # sweep, matching phase 13b's 02_extract_base_features.py precomputation pattern.
    print(f"[{label}] Precomputing projected base features (model.encode_feature) for "
          f"{len(item_ids)} catalog items...")
    with torch.no_grad():
        proj_chunks = []
        CHUNK = 8192
        for start in range(0, len(item_ids), CHUNK):
            chunk = torch.tensor(raw_embeddings[start:start + CHUNK], device=DEVICE)
            proj_chunks.append(model.encode_feature(chunk).cpu().numpy())
        base_features = np.concatenate(proj_chunks, axis=0)

    rng = np.random.default_rng(SEED)
    diag_qidxs = rng.choice(len(queries), size=min(DIAG_SAMPLE_SIZE, len(queries)), replace=False).tolist()
    rng2 = np.random.default_rng(SEED + 1)
    overlap_qidxs = rng2.choice(len(queries), size=min(OVERLAP_SAMPLE_SIZE, len(queries)), replace=False).tolist()

    print(f"[{label}] Computing raw-SigLIP reference top-K for overlap diagnostic...")
    raw_topk = raw_siglip_topk(pools, queries, item_ids, raw_embeddings, idx, overlap_qidxs)

    results = []
    alpha0_topk_diag = None
    topk_cache = {}  # alpha -> {qi: [item_ids]}, overlap_qidxs only, for the adjacent-vs-distant smoothness check
    for alpha in ALPHAS:
        recall, n_total, n_skipped, diag_topk = evaluate_alpha(
            model, alpha, pools, queries, item_ids, base_features, raw_embeddings,
            categories, item_cat_lookup, idx, diag_qidxs, overlap_qidxs)
        axis1, axis2 = diagnostics(queries, raw_embeddings, idx, diag_qidxs, diag_topk)
        overlap_raw = overlap(diag_topk, raw_topk, overlap_qidxs)
        if alpha == 0.0:
            alpha0_topk_diag = dict(diag_topk)
        overlap_alpha0 = overlap(diag_topk, alpha0_topk_diag, overlap_qidxs) if alpha0_topk_diag else 1.0
        topk_cache[str(alpha)] = {str(qi): diag_topk[qi] for qi in overlap_qidxs if qi in diag_topk}

        row = {
            "alpha": alpha, "recall": recall, "n_total": n_total, "n_skipped": n_skipped,
            "axis1_visual_sim": axis1, "axis2_hit_rate": axis2,
            "overlap_with_raw": overlap_raw, "overlap_with_alpha0": overlap_alpha0,
        }
        results.append(row)
        print(f"[{label}] alpha={alpha}: recall={recall} axis1={axis1:.4f} axis2={axis2:.4f} "
              f"overlap_raw={overlap_raw:.4f} overlap_a0={overlap_alpha0:.4f} "
              f"(n={n_total}, skipped={n_skipped})")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{label}] Saved {out_json}")

    topk_cache_path = out_json.parent / f"topk_cache_{label}.json"
    with open(topk_cache_path, "w") as f:
        json.dump(topk_cache, f)
    print(f"[{label}] Saved {topk_cache_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alphas", default=None, help="comma-separated override, e.g. '0.0' for a timing test")
    args = parser.parse_args()
    if args.alphas:
        ALPHAS = [float(a) for a in args.alphas.split(",")]
    run_sweep(Path(args.checkpoint), args.label, Path(args.out))

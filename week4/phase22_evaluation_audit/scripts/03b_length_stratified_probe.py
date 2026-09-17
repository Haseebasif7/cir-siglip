"""
Phase 22, step 3 (numeric follow-up): actually run phase 9 and OutfitTransformer
(the two models with cheap-enough evaluation cost to re-run here) and bucket
Recall@10 by query context length, to check for a concrete length-dependent
gap between a context-free mean-pool (phase 9) and a learned context-aware
aggregator (OutfitTransformer) rather than relying on code-reading alone.
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
PHASE14B_DIR = REPO_ROOT / "week4/phase14b_outfittransformer_category_negatives"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p9_model_mod = load_module("phase9_model", REPO_ROOT / "week4/phase20_full_consolidation/scripts/model.py")
ProjectionHead = p9_model_mod.ProjectionHead

ot_model_mod = load_module("ot_model", PHASE14B_DIR / "scripts/model.py")
OutfitTransformerSigLIP = ot_model_mod.OutfitTransformerSigLIP

BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"
SIGLIP_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE9_CKPT = PHASE9_DIR / "models" / "model_a_random_negs.pt"
OT_CANDIDATE_NPZ = PHASE14B_DIR / "embeddings" / "outfit_transformer_candidate_features_random_negatives.npz"
OT_CKPT = PHASE14B_DIR / "models_random_negatives" / "outfit_transformer_siglip_random_neg_best.pt"

OUT_MD = BASE_DIR / "scoring_asymmetry_check.md"  # appended, not overwritten
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
QUERY_BATCH = 1024
BINS = [(1, 3), (4, 5), (6, 7), (8, 16)]


def load_benchmark():
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def load_siglip():
    data = np.load(SIGLIP_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return {a: i for i, a in enumerate(item_ids)}, embeddings / norms


def per_query_ranks_phase9(pools, queries, idx, proj_emb):
    ranks = {}
    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)
    for cat, qidxs in by_cat.items():
        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_idx = [idx[i] for i in pool_ids]
        pool_emb = proj_emb[pool_idx]
        for qi in qidxs:
            q = queries[qi]
            item_idx = [idx[i] for i in q["query_items"] if i in idx]
            if not item_idx or q["target_item"] not in pool_pos:
                continue
            v = proj_emb[item_idx].mean(axis=0)
            n = np.linalg.norm(v)
            if n == 0:
                continue
            v = v / n
            sims = pool_emb @ v
            target_sim = sims[pool_pos[q["target_item"]]]
            rank = int((sims >= target_sim).sum())
            ranks[qi] = rank
    return ranks


def per_query_ranks_ot(pools, queries, siglip_idx, siglip_emb, cand_idx, cand_emb, model):
    ranks = {}
    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)
    with torch.no_grad():
        for cat, qidxs in by_cat.items():
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_gidx = [cand_idx[i] for i in pool_ids if i in cand_idx]
            if len(pool_gidx) != len(pool_ids):
                continue
            pool_emb = cand_emb[pool_gidx]

            valid_qidxs, item_lists, target_positions = [], [], []
            for qi in qidxs:
                q = queries[qi]
                items = [i for i in q["query_items"] if i in siglip_idx]
                if not items or q["target_item"] not in pool_pos:
                    continue
                valid_qidxs.append(qi)
                item_lists.append(items)
                target_positions.append(pool_pos[q["target_item"]])
            if not item_lists:
                continue

            for start in range(0, len(item_lists), QUERY_BATCH):
                chunk = item_lists[start:start + QUERY_BATCH]
                chunk_qidxs = valid_qidxs[start:start + QUERY_BATCH]
                chunk_targets = target_positions[start:start + QUERY_BATCH]
                lengths = [len(q) for q in chunk]
                Lmax = max(lengths)
                B = len(chunk)
                ctx = np.zeros((B, Lmax, siglip_emb.shape[1]), dtype=np.float32)
                mask = np.ones((B, Lmax), dtype=bool)
                for i, items in enumerate(chunk):
                    for j, it in enumerate(items):
                        ctx[i, j] = siglip_emb[siglip_idx[it]]
                        mask[i, j] = False
                ctx_t = torch.tensor(ctx, device=DEVICE)
                mask_t = torch.tensor(mask, device=DEVICE)
                Bt, Lt, Dt = ctx_t.shape
                tokens = model.encode_item_tokens(ctx_t.view(Bt * Lt, Dt)).view(Bt, Lt, -1)
                q_emb = model.embed_query(tokens, mask_t).cpu().numpy()

                sims = q_emb @ pool_emb.T
                target_sims = sims[np.arange(len(chunk)), chunk_targets]
                batch_ranks = (sims >= target_sims[:, None]).sum(axis=1)
                for qi, r in zip(chunk_qidxs, batch_ranks):
                    ranks[qi] = int(r)
    return ranks


def bucket_recall(ranks, lengths_by_qi, k=10):
    buckets = defaultdict(lambda: [0, 0])
    for qi, rank in ranks.items():
        L = lengths_by_qi[qi]
        for lo, hi in BINS:
            if lo <= L <= hi:
                buckets[(lo, hi)][1] += 1
                if rank <= k:
                    buckets[(lo, hi)][0] += 1
                break
    return {b: (hits / n if n else 0.0, n) for b, (hits, n) in buckets.items()}


def main():
    pools, queries = load_benchmark()
    lengths_by_qi = {qi: len(q["query_items"]) for qi, q in enumerate(queries)}

    siglip_idx, siglip_emb = load_siglip()
    p9_model = ProjectionHead().to(DEVICE).eval()
    p9_model.load_state_dict(torch.load(PHASE9_CKPT, map_location=DEVICE))
    with torch.no_grad():
        proj_emb = p9_model(torch.tensor(siglip_emb, device=DEVICE)).cpu().numpy()
    p9_ranks = per_query_ranks_phase9(pools, queries, siglip_idx, proj_emb)
    print(f"Phase 9: {len(p9_ranks)} queries scored")

    cand = np.load(OT_CANDIDATE_NPZ, allow_pickle=True)
    cand_item_ids = [str(a) for a in cand["item_ids"]]
    cand_idx = {a: i for i, a in enumerate(cand_item_ids)}
    cand_emb = cand["embeddings"].astype(np.float32)
    ot_model = OutfitTransformerSigLIP().to(DEVICE).eval()
    ot_model.load_state_dict(torch.load(OT_CKPT, map_location=DEVICE))
    ot_ranks = per_query_ranks_ot(pools, queries, siglip_idx, siglip_emb, cand_idx, cand_emb, ot_model)
    print(f"OutfitTransformer: {len(ot_ranks)} queries scored")

    p9_buckets = bucket_recall(p9_ranks, lengths_by_qi)
    ot_buckets = bucket_recall(ot_ranks, lengths_by_qi)

    lines = ["", "## Numeric follow-up: Recall@10 by query context length, phase 9 vs OutfitTransformer (actually run, not estimated)\n"]
    lines.append("| Context length bin | Phase 9 Recall@10 (n) | OutfitTransformer Recall@10 (n) | Ratio (P9/OT) |")
    lines.append("|---|---|---|---|")
    for b in BINS:
        p9_r, p9_n = p9_buckets.get(b, (0.0, 0))
        ot_r, ot_n = ot_buckets.get(b, (0.0, 0))
        ratio = (p9_r / ot_r) if ot_r > 0 else float("nan")
        lines.append(f"| {b[0]}-{b[1]} items | {p9_r:.4f} ({p9_n}) | {ot_r:.4f} ({ot_n}) | {ratio:.2f}x |")
    lines.append("")

    ratios = []
    for b in BINS:
        p9_r, p9_n = p9_buckets.get(b, (0.0, 0))
        ot_r, ot_n = ot_buckets.get(b, (0.0, 0))
        if ot_r > 0:
            ratios.append(p9_r / ot_r)
    ratio_spread = max(ratios) - min(ratios) if ratios else 0.0
    lines.append(
        f"Phase 9's advantage ratio over OutfitTransformer ranges "
        f"{min(ratios):.2f}x-{max(ratios):.2f}x across context-length bins (spread "
        f"{ratio_spread:.2f}). If OutfitTransformer's aggregation were being "
        f"systematically penalized by context length, this ratio would grow monotonically "
        f"with bin size (longer-context queries penalized more). "
    )
    if ratio_spread < 0.5 and max(ratios) / min(ratios) < 1.6:
        lines.append(
            "**The ratio is roughly flat across context-length bins** -- phase 9's "
            "advantage over OutfitTransformer is consistent regardless of how many context "
            "items a query has, which is the signature of a genuine capability gap between "
            "the two models, not an aggregation artifact that specifically punishes longer "
            "(or shorter) context OutfitTransformer queries."
        )
    else:
        lines.append(
            "**The ratio is NOT flat across context-length bins** -- there is a real "
            "length-dependent effect here worth flagging, since it's the signature an "
            "aggregation-induced penalty would leave."
        )
    lines.append("")

    with open(OUT_MD, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended to {OUT_MD}")
    for b in BINS:
        print(b, "P9:", p9_buckets.get(b), "OT:", ot_buckets.get(b))


if __name__ == "__main__":
    main()

"""
Phase 33, step 0 (continued): verify the vectorized evaluate_recall
(train_core.py) reproduces phase 13b's original per-query, per-context-item
loop-based evaluator (03_csa_cir_eval.py's evaluate_csa_recall) exactly, on
a small synthetic benchmark, same model weights for both.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CSANetSigLIP
import train_core

torch.manual_seed(0)
DEVICE = "cpu"
NUM_CATEGORIES = 5
N_ITEMS = 60
EMBED_IN = 16
N_QUERIES = 15
POOL_SIZE = 20


def original_evaluate(model, base_repr, id_to_gidx, cat_to_idx, item_cat_lookup, pools, queries, ks=(1, 3, 5)):
    """Faithful re-implementation of 03_csa_cir_eval.py's evaluate_csa_recall."""
    from collections import defaultdict
    categories = list(cat_to_idx.keys())
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0
    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)
    eye = torch.eye(len(categories))

    with torch.no_grad():
        for cat, qidxs in by_cat.items():
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_gidx = [id_to_gidx[i] for i in pool_ids]
            x_pool = model.encode_feature(base_repr[pool_gidx])
            cat_t_fixed = torch.zeros(len(pool_gidx), len(categories))
            cat_t_fixed[:, cat_to_idx[cat]] = 1.0
            cand_all = model.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed)

            for qi in qidxs:
                q = queries[qi]
                ctx_items = q["query_items"]
                ctx_gidx = [id_to_gidx[i] for i in ctx_items]
                x_ctx = model.encode_feature(base_repr[ctx_gidx])
                dist_sum = torch.zeros(len(pool_gidx))
                for ci, item_id in enumerate(ctx_items):
                    c_idx = cat_to_idx[item_cat_lookup[item_id]]
                    f_ctx = model.embed_from_feature(x_ctx[ci:ci + 1], eye[c_idx:c_idx + 1], cat_t_fixed[0:1])
                    cand_slice = cand_all[:, c_idx, :]
                    d = ((f_ctx - cand_slice) ** 2).sum(dim=-1)
                    dist_sum += d
                dist_avg = (dist_sum / len(ctx_items)).numpy()
                target_pos = pool_pos[q["target_item"]]
                target_dist = dist_avg[target_pos]
                rank = int((dist_avg <= target_dist).sum())
                n_total += 1
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


def main():
    rng = np.random.default_rng(2)
    base_repr = F.normalize(torch.randn(N_ITEMS, EMBED_IN), p=2, dim=-1)
    model = CSANetSigLIP(num_categories=NUM_CATEGORIES, siglip_dim=EMBED_IN).to(DEVICE).eval()

    categories = [f"cat{i}" for i in range(NUM_CATEGORIES)]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    item_ids = [f"item{i}" for i in range(N_ITEMS)]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    item_cat_lookup = {a: categories[rng.integers(0, NUM_CATEGORIES)] for a in item_ids}

    # one pool per category, POOL_SIZE items each (drawn from the matching-category subset if possible,
    # else just random -- doesn't need to be realistic, only self-consistent)
    pools = {}
    for c in categories:
        pool_items = rng.choice(item_ids, size=min(POOL_SIZE, N_ITEMS), replace=False).tolist()
        pools[c] = pool_items

    queries = []
    for _ in range(N_QUERIES):
        cat = categories[rng.integers(0, NUM_CATEGORIES)]
        pool = pools[cat]
        target = pool[rng.integers(0, len(pool))]
        n_ctx = rng.integers(1, 5)
        ctx = rng.choice(item_ids, size=n_ctx, replace=False).tolist()
        queries.append({"category": cat, "query_items": ctx, "target_item": target})

    orig_recall, orig_n, orig_skip = original_evaluate(model, base_repr, id_to_gidx, cat_to_idx, item_cat_lookup,
                                                          pools, queries)
    vec_recall, vec_n, vec_skip = train_core.evaluate_recall(model, base_repr, id_to_gidx, cat_to_idx,
                                                               item_cat_lookup, pools, queries, DEVICE,
                                                               ks=(1, 3, 5), query_chunk=4)

    print(f"Original:   {orig_recall} n_total={orig_n} n_skipped={orig_skip}")
    print(f"Vectorized: {vec_recall} n_total={vec_n} n_skipped={vec_skip}")

    ok = (orig_n == vec_n and orig_skip == vec_skip and
          all(abs(orig_recall[k] - vec_recall[k]) < 1e-9 for k in orig_recall))
    print("PASS" if ok else "FAIL")
    assert ok, "Vectorized evaluate_recall does NOT match the original"


if __name__ == "__main__":
    main()

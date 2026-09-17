"""
Phase 12b: identical copy of phase 12's shared CIR benchmark loading +
Recall@K evaluator (`week4/phase12_controllable_modes/scripts/cir_eval.py`),
except BENCHMARK_JSON points directly at phase 12's original
`cir_benchmark.json` rather than duplicating that file here -- the brief for
this phase explicitly asks to reuse phase 12's harness directly, and pointing
at the same file (rather than copying it) guarantees phase 12b is scored
against the exact same queries and candidate pools with no risk of drift.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE12_DIR = BASE_DIR.parent / "phase12_controllable_modes"
BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"


def load_benchmark():
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def _query_vector(embeddings, idx, query_items):
    item_idx = [idx[i] for i in query_items if i in idx]
    if not item_idx:
        return None
    v = embeddings[item_idx].mean(axis=0)
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    return v / norm


def evaluate_recall(pools, queries, item_ids, embeddings, ks=(10, 30, 50)):
    """embeddings: (N, D) array aligned with item_ids (any consistent norm --
    ranking only depends on relative cosine similarity, so raw SigLIP's
    L2-normalized vectors and a projection head's L2-normalized outputs both
    work directly as dot-product similarity)."""
    idx = {a: i for i, a in enumerate(item_ids)}
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    for cat, qidxs in by_cat.items():
        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_idx = [idx[i] for i in pool_ids]
        pool_emb = embeddings[pool_idx]  # (P, D)

        query_vecs, target_positions = [], []
        for qi in qidxs:
            q = queries[qi]
            qv = _query_vector(embeddings, idx, q["query_items"])
            if qv is None or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            query_vecs.append(qv)
            target_positions.append(pool_pos[q["target_item"]])
        if not query_vecs:
            continue

        query_mat = np.stack(query_vecs)  # (Q, D)
        sims = query_mat @ pool_emb.T  # (Q, P)
        target_positions = np.array(target_positions)
        target_sims = sims[np.arange(len(query_vecs)), target_positions]
        ranks = (sims >= target_sims[:, None]).sum(axis=1)  # 1-indexed, ties count as ahead

        n_total += len(query_vecs)
        for k in ks:
            hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


def topk_for_query(pools, query, item_ids, embeddings, k=5):
    """Single-query helper for qualitative/diagnostic scripts: returns the
    top-k candidate item_ids from the query's own category pool, ranked by
    similarity, plus their scores."""
    idx = {a: i for i, a in enumerate(item_ids)}
    pool_ids = pools[query["category"]]
    pool_idx = [idx[i] for i in pool_ids]
    pool_emb = embeddings[pool_idx]

    qv = _query_vector(embeddings, idx, query["query_items"])
    if qv is None:
        return [], []
    sims = pool_emb @ qv
    order = np.argsort(-sims)[:k]
    top_ids = [pool_ids[i] for i in order]
    top_sims = [float(sims[i]) for i in order]
    return top_ids, top_sims

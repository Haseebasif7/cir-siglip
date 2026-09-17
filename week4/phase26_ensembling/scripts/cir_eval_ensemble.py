"""Phase 26: CIR benchmark evaluator generalized to score-averaging
ensembles. Single-model evaluation is the len(embeddings_list)==1 case of
the same function, so this replaces cir_eval.py's evaluate_recall for this
phase rather than duplicating it.

Deliberate design point (per the brief): ensembling here averages
SIMILARITY SCORES across members, never raw embedding vectors. Independently
initialized/trained models have no shared coordinate system, so each
member's own (query_vector, pool_matrix) similarity matrix is computed in
that member's own embedding space first, and only the resulting per-pair
cosine-similarity scores are averaged across members before ranking.
"""
import json
from collections import defaultdict

import numpy as np


def load_benchmark(path):
    with open(path) as f:
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


def evaluate_recall_ensemble(pools, queries, item_ids, embeddings_list, ks=(10, 30, 50)):
    """embeddings_list: list of (N_items, D) arrays, one per ensemble member,
    each already L2-normalized per-row and each in that member's own space.
    A single-element list reproduces plain single-model evaluate_recall."""
    idx = {a: i for i, a in enumerate(item_ids)}
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    for cat, qidxs in by_cat.items():
        if cat not in pools:
            n_skipped += len(qidxs)
            continue
        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_idx = [idx[i] for i in pool_ids]

        # per-member query vectors + skip decisions must agree across members
        # (they use the same idx/pool regardless of embedding space), so
        # skip logic is computed once using member 0 and applied to all.
        query_vecs_per_member = [[] for _ in embeddings_list]
        target_positions = []
        qidx_kept = []
        for qi in qidxs:
            q = queries[qi]
            qv0 = _query_vector(embeddings_list[0], idx, q["query_items"])
            if qv0 is None or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            qidx_kept.append(qi)
            target_positions.append(pool_pos[q["target_item"]])
        if not qidx_kept:
            continue

        sims_sum = None
        for m, emb in enumerate(embeddings_list):
            pool_emb = emb[pool_idx]
            qvecs = []
            for qi in qidx_kept:
                q = queries[qi]
                qvecs.append(_query_vector(emb, idx, q["query_items"]))
            query_mat = np.stack(qvecs)
            sims = query_mat @ pool_emb.T
            sims_sum = sims if sims_sum is None else sims_sum + sims
        sims_avg = sims_sum / len(embeddings_list)

        target_positions = np.array(target_positions)
        target_sims = sims_avg[np.arange(len(qidx_kept)), target_positions]
        ranks = (sims_avg >= target_sims[:, None]).sum(axis=1)

        n_total += len(qidx_kept)
        for k in ks:
            hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped

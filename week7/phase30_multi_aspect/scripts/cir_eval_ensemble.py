"""
Phase 30: CIR benchmark evaluator for score-averaging ensembles of the
multi-aspect architecture. Adapted from phase 28's own
cir_eval_ensemble.py -- same score-averaging principle (average per-member
SCORES, never raw embedding vectors, since independently-trained models
share no coordinate system), generalized so each member's "score" is itself
the weighted-sum-of-per-aspect-cosines defined in model.py, not a single dot
product.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MultiAspectProjectionHead, inbatch_scores


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def build_base_repr(image_emb, text_emb, device):
    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    return F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)


def project_all_aspects(ckpt_path, base_repr_t, device):
    model = MultiAspectProjectionHead().to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    with torch.no_grad():
        proj = model(base_repr_t)  # (n_items, K, D)
    return model, proj


def _query_aspects(proj_aspects, idx, query_items, device):
    item_idx = [idx[i] for i in query_items if i in idx]
    if not item_idx:
        return None
    v = proj_aspects[item_idx].mean(dim=0)
    v = F.normalize(v, p=2, dim=-1)
    return v


def evaluate_recall_ensemble_multiaspect(pools, queries, item_ids, members, device, ks=(10, 30, 50)):
    """members: list of (model, proj_aspects) tuples, one per ensemble member
    -- proj_aspects is that member's own (n_items, K, D) projected aspect
    tensor, model is needed for its own agg_weights layer (query-specific,
    not shared across members)."""
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
        pool_idx = [idx[i] for i in pool_ids if i in idx]

        qidx_kept, target_positions, ctx_lists = [], [], []
        for qi in qidxs:
            q = queries[qi]
            item_idx = [idx[i] for i in q["query_items"] if i in idx]
            if not item_idx or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            qidx_kept.append(qi)
            target_positions.append(pool_pos[q["target_item"]])
            ctx_lists.append(item_idx)
        if not qidx_kept:
            continue

        sims_sum = None
        for model, proj_aspects in members:
            pool_aspects = proj_aspects[pool_idx]  # (P, K, D)
            q_aspects_list = [F.normalize(proj_aspects[ctx].mean(dim=0), p=2, dim=-1) for ctx in ctx_lists]
            q_aspects = torch.stack(q_aspects_list)  # (nq, K, D)
            with torch.no_grad():
                q_weights = model.agg_weights(q_aspects)  # (nq, K)
                sims = inbatch_scores(q_aspects, q_weights, pool_aspects).cpu().numpy()  # (nq, P)
            sims_sum = sims if sims_sum is None else sims_sum + sims
        sims_avg = sims_sum / len(members)

        target_positions_arr = np.array(target_positions)
        target_sims = sims_avg[np.arange(len(qidx_kept)), target_positions_arr]
        ranks = (sims_avg >= target_sims[:, None]).sum(axis=1)

        n_total += len(qidx_kept)
        for k in ks:
            hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped

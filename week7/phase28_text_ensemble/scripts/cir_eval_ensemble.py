"""
Phase 28: CIR benchmark evaluator for score-averaging ensembles of phase 27's
text-only architecture. Adapted from phase 26's
week4/phase26_ensembling/scripts/cir_eval_ensemble.py -- identical math
(average per-member cosine SIMILARITY SCORES, never raw embedding vectors,
since independently-trained models share no coordinate system), the only
difference is each member's raw per-item representation is base_repr =
normalize(concat(image, text)), 1536-d, matching phase 27's text_only
mechanism, instead of phase 26's plain 768-d image embedding.
"""
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def build_base_repr(image_emb, text_emb, device):
    """image_emb, text_emb: (N, 768) numpy arrays, image already L2-normalized.
    Returns a (N, 1536) torch tensor, L2-normalized -- exactly modal_app.py's
    own construction, reproduced locally so evaluation isn't dependent on a
    live Modal call."""
    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    return F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)


class ProjectionHeadGeneral(nn.Module):
    def __init__(self, in_dim=1536, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def project_all(ckpt_path, base_repr_t, device, hidden_dims=(1024,), out_dim=128):
    model = ProjectionHeadGeneral(in_dim=base_repr_t.shape[1], hidden_dims=hidden_dims,
                                   out_dim=out_dim).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    with torch.no_grad():
        proj = model(base_repr_t).cpu().numpy()
    return proj


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
    each ALREADY the member's own post-projection embeddings (128-d),
    L2-normalized per row. A single-element list reproduces solo evaluation.
    Post-projection pooling throughout -- identical mechanism to phase 27's
    text_only (use_category="none") eval, itself identical to week 6's own."""
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

        qidx_kept, target_positions = [], []
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
        for emb in embeddings_list:
            pool_emb = emb[pool_idx]
            qvecs = [_query_vector(emb, idx, queries[qi]["query_items"]) for qi in qidx_kept]
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

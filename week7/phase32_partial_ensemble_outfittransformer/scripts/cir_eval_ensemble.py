"""
Phase 32: CIR benchmark evaluator for score-averaging ensembles of phase
31's strengthened OutfitTransformer. Adapted from phase 31's own
07_final_test_eval.py evaluator (embed_query/embed_item_alone split), the
only change is combining multiple members via score averaging, identical
principle to phases 26/28/30: average per-member SIMILARITY SCORES, never
raw embedding vectors, since independently-trained models share no
coordinate system.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP

MODEL_KWARGS = dict(siglip_dim=1536, d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512, dropout=0.1)
QUERY_BATCH = 1024


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def build_base_repr(image_emb, text_emb, device):
    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    return F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)


def load_member(ckpt_path, device):
    model = OutfitTransformerSigLIP(**MODEL_KWARGS).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def precompute_candidates(model, base_repr, device):
    outs = []
    with torch.no_grad():
        for start in range(0, base_repr.shape[0], 4096):
            chunk = base_repr[start:start + 4096]
            tokens = model.encode_item_tokens(chunk)
            outs.append(model.embed_item_alone(tokens).cpu().numpy())
    return np.concatenate(outs, axis=0)


def compute_query_embeddings(model, base_repr, idx, query_item_lists, device):
    outs = []
    for start in range(0, len(query_item_lists), QUERY_BATCH):
        chunk = query_item_lists[start:start + QUERY_BATCH]
        Lmax = max(len(q) for q in chunk)
        B = len(chunk)
        ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
        mask = np.ones((B, Lmax), dtype=bool)
        for i, items in enumerate(chunk):
            for j, it in enumerate(items):
                ctx_gidx[i, j] = idx[it]
                mask[i, j] = False
        ctx_t = base_repr[torch.tensor(ctx_gidx, device=device)]
        mask_t = torch.tensor(mask, device=device)
        with torch.no_grad():
            B_, L_, D_ = ctx_t.shape
            tokens = model.encode_item_tokens(ctx_t.reshape(B_ * L_, D_)).reshape(B_, L_, -1)
            q_emb = model.embed_query(tokens, mask_t)
        outs.append(q_emb.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_recall_ensemble(pools, queries, idx, base_repr, members, device, ks=(10, 30, 50)):
    """members: list of (model, candidate_emb) tuples, one per ensemble
    member -- candidate_emb is that member's own precomputed (n_items,
    d_embed) embed_item_alone projection. A single-element list reproduces
    solo evaluation."""
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

        query_item_lists, target_positions = [], []
        for qi in qidxs:
            q = queries[qi]
            items = [i for i in q["query_items"] if i in idx]
            if not items or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            query_item_lists.append(items)
            target_positions.append(pool_pos[q["target_item"]])
        if not query_item_lists:
            continue

        sims_sum = None
        for model, candidate_emb in members:
            pool_emb = candidate_emb[pool_idx]
            query_mat = compute_query_embeddings(model, base_repr, idx, query_item_lists, device)
            sims = query_mat @ pool_emb.T
            sims_sum = sims if sims_sum is None else sims_sum + sims
        sims_avg = sims_sum / len(members)

        target_positions_arr = np.array(target_positions)
        target_sims = sims_avg[np.arange(len(query_item_lists)), target_positions_arr]
        ranks = (sims_avg >= target_sims[:, None]).sum(axis=1)

        n_total += len(query_item_lists)
        for k in ks:
            hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped

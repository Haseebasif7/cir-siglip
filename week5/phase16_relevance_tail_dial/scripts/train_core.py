"""Shared data loading, batch construction, and loss-balancing helpers for
phase 16's relevance/tail-exposure dial, imported by both 02_smoke_test.py
and 03_train.py -- matching the train_core.py sharing convention phases
13+/15 established (rather than phase 12c's single-script pattern), since
phase 16 needs the exact same batch/loss logic in two separate scripts and
duplicating it risks silent drift between them.

Batch construction mirrors phase 12c directly: one shared batch of positive
also_buy edges per step, R_NEG=8 random negatives per anchor (this project's
own phase 7-9 finding is that mined hard negatives underperform for this
loss, and phase 12c's complement mode standardized on random negatives for
the same reason). The one addition beyond phase 12c: each edge also carries
a precomputed IPS weight (from 01_prepare_training_data.py) used only by the
tail-exposure loss.
"""
import random
from pathlib import Path

import numpy as np
import torch

from model import ControllableProjectionHead, mean_pairwise_cosine, mnrl_loss, mnrl_loss_weighted, uniformity_loss

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_NPZ = BASE_DIR / "data" / "training_data_with_weights.npz"
POSITIVE_SETS_JSON = BASE_DIR / "data" / "positive_sets.json"

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU = 0.07
R_NEG = 8
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-4
N_CALIBRATION_BATCHES = 20
UNIFORMITY_WEIGHT = 1.0  # only applied if enabled by the smoke test
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Edge:
    __slots__ = ("source", "target", "weight")

    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight


def load_data():
    import json

    d = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in d["asins"]]
    embeddings = d["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    t = np.load(TRAINING_DATA_NPZ, allow_pickle=True)
    sources = t["source"]
    targets = t["target"]
    splits = t["split"]
    weights = t["weight"]

    train_edges, val_edges = [], []
    for s, tg, sp, w in zip(sources, targets, splits, weights):
        e = Edge(str(s), str(tg), float(w))
        if sp == "train":
            train_edges.append(e)
        else:
            val_edges.append(e)

    with open(POSITIVE_SETS_JSON) as f:
        raw_positive_sets = json.load(f)
    all_positive_targets = {k: set(v) for k, v in raw_positive_sets.items()}

    return embeddings, idx, item_ids, train_edges, val_edges, all_positive_targets


def sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_item, k):
    negs = []
    exclude = positive_set | {anchor}
    tries, max_tries = 0, k * 20
    while len(negs) < k and tries < max_tries:
        cand = idx_to_item[rng.randrange(n_items)]
        if cand not in exclude and cand not in negs:
            negs.append(cand)
        tries += 1
    while len(negs) < k:
        cand = idx_to_item[rng.randrange(n_items)]
        negs.append(cand)
    return negs


def build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, k):
    n_items = len(item_ids)
    out = []
    for e in batch_edges:
        positive_set = all_positive_targets.get(e.source, set())
        out.append(sample_random_negatives(rng, e.source, positive_set, n_items, item_ids, k))
    return out


def build_inbatch_mask(batch_edges, all_positive_targets):
    B = len(batch_edges)
    mask = np.zeros((B, B), dtype=bool)
    for i, ei in enumerate(batch_edges):
        positives_i = all_positive_targets.get(ei.source, set())
        for j, ej in enumerate(batch_edges):
            if i != j and ej.target in positives_i:
                mask[i, j] = True
    return mask


def _project_batch(model, embeddings, idx, batch_edges, neg_lists, alpha):
    anchors = [e.source for e in batch_edges]
    positives = [e.target for e in batch_edges]
    a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
    p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
    extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
    extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

    z_a = model(a_emb, alpha=alpha)
    z_p = model(p_emb, alpha=alpha)
    B, K, D = extra_emb.shape
    z_extra = model(extra_emb.reshape(B * K, D), alpha=alpha).reshape(B, K, -1)
    return z_a, z_p, z_extra


def relevance_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets):
    z_a, z_p, z_extra = _project_batch(model, embeddings, idx, batch_edges, neg_lists, alpha=1.0)
    mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
    return mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU), z_a


def tail_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets):
    z_a, z_p, z_extra = _project_batch(model, embeddings, idx, batch_edges, neg_lists, alpha=0.0)
    mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
    weights = torch.tensor([e.weight for e in batch_edges], dtype=torch.float32, device=DEVICE)
    return mnrl_loss_weighted(z_a, z_p, z_extra, mask, weights, tau=TAU), z_a


def combined_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets,
                             weight_tail, uniformity_enabled=False):
    rel_loss, z_a_rel = relevance_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    tail_loss, z_a_tail = tail_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    total = rel_loss + weight_tail * tail_loss
    if uniformity_enabled:
        total = total + UNIFORMITY_WEIGHT * uniformity_loss(z_a_rel) + UNIFORMITY_WEIGHT * uniformity_loss(z_a_tail)
    return total, rel_loss, tail_loss


def measure_initial_magnitudes(model, embeddings, idx, edges, all_positive_targets, item_ids, rng, n_batches):
    model.eval()
    order = list(range(len(edges)))
    rng.shuffle(order)
    rel_vals, tail_vals = [], []
    with torch.no_grad():
        for b in range(n_batches):
            batch_idx = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            batch_edges = [edges[i] for i in batch_idx]
            neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)
            rel_loss, _ = relevance_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
            tail_loss, _ = tail_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
            rel_vals.append(rel_loss.item())
            tail_vals.append(tail_loss.item())
    return float(np.mean(rel_vals)), float(np.mean(tail_vals))


def grad_norm_check(model, embeddings, idx, edges, all_positive_targets, item_ids, rng, weight_tail):
    order = list(range(len(edges)))
    rng.shuffle(order)
    batch_edges = [edges[i] for i in order[:BATCH_SIZE]]
    neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

    def net_grad_norm(loss):
        model.zero_grad()
        loss.backward()
        total = 0.0
        for p in model.net.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    model.train()
    rel_loss, _ = relevance_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    rel_grad_norm = net_grad_norm(rel_loss)

    tail_loss, _ = tail_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    tail_grad_norm_unweighted = net_grad_norm(tail_loss)

    tail_loss2, _ = tail_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
    tail_grad_norm_weighted = net_grad_norm(weight_tail * tail_loss2)

    model.zero_grad()
    return rel_grad_norm, tail_grad_norm_unweighted, tail_grad_norm_weighted


def run_epoch_train(model, optimizer, embeddings, idx, edges, all_positive_targets, item_ids, rng,
                     weight_tail, uniformity_enabled=False, log_every=2000):
    model.train()
    order = list(range(len(edges)))
    rng.shuffle(order)
    total_rel, total_tail, n_batches = 0.0, 0.0, 0

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue
        batch_edges = [edges[i] for i in batch_idx]
        neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

        loss, rel_loss, tail_loss = combined_loss_for_batch(
            model, embeddings, idx, batch_edges, neg_lists, all_positive_targets, weight_tail, uniformity_enabled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_rel += rel_loss.item()
        total_tail += tail_loss.item()
        n_batches += 1
        if log_every and n_batches % log_every == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, "
                  f"relevance={total_rel/n_batches:.4f} tail={total_tail/n_batches:.4f}")

    return total_rel / max(n_batches, 1), total_tail / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets):
    model.eval()
    total_rel, total_tail, n_batches = 0.0, 0.0, 0
    for start in range(0, len(val_edges), BATCH_SIZE):
        batch_edges = val_edges[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_val_negs[start:start + BATCH_SIZE]
        rel_loss, _ = relevance_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
        tail_loss, _ = tail_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets)
        total_rel += rel_loss.item()
        total_tail += tail_loss.item()
        n_batches += 1
    return total_rel / max(n_batches, 1), total_tail / max(n_batches, 1)

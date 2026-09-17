"""Shared data loading, batch construction, and loss-balancing helpers for
phase 16c, imported by both 04_smoke_test.py and 05_train.py -- same
train_core.py sharing convention phase 16 used.

Structural difference from phase 16: relevance mode and tail-exposure mode
now train on two DIFFERENT edge populations (phase 7's full 76,293 also_buy
edges vs. this phase's 37,922 attribute-based pairs), not one shared batch
of anchors reused for both losses. Each training step draws two
INDEPENDENT batches -- one from each population. One epoch is defined as
one full pass over the LARGER population (relevance/also_buy); the smaller
population (tail/attribute) is drawn from a cyclic iterator that reshuffles
and restarts whenever it runs out, so every step still gets a fresh batch
even though its source pool is smaller.
"""
import random
from pathlib import Path

import numpy as np
import torch

from model import ControllableProjectionHead, mean_pairwise_cosine, mnrl_loss, uniformity_loss

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"
ALSO_BUY_EDGES_JSON = PHASE7_DIR / "data" / "positive_edges.json"
ATTRIBUTE_EDGES_JSON = BASE_DIR / "data" / "attribute_pairs.json"

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
UNIFORMITY_WEIGHT = 1.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Edge:
    __slots__ = ("source", "target")

    def __init__(self, source, target):
        self.source = source
        self.target = target


def _load_edges(path):
    import json
    with open(path) as f:
        records = json.load(f)
    train = [Edge(e["source"], e["target"]) for e in records if e["split"] == "train"]
    val = [Edge(e["source"], e["target"]) for e in records if e["split"] == "val"]
    return train, val


def load_data():
    d = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in d["asins"]]
    embeddings = d["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    rel_train, rel_val = _load_edges(ALSO_BUY_EDGES_JSON)
    tail_train, tail_val = _load_edges(ATTRIBUTE_EDGES_JSON)

    # false-negative exclusion set built from BOTH edge sources combined (conservative:
    # exclude anything that's a known positive from either signal), matching phase 8's
    # own asymmetry precedent (positive TRAINING edges restricted, exclusion set broader).
    all_positive_targets = {}
    for e in rel_train + rel_val + tail_train + tail_val:
        all_positive_targets.setdefault(e.source, set()).add(e.target)

    return embeddings, idx, item_ids, rel_train, rel_val, tail_train, tail_val, all_positive_targets


class CyclicSampler:
    """Reshuffles and restarts when exhausted -- used for the smaller
    (attribute) edge population so every training step still gets a full
    batch even though its source pool is smaller than the relevance one."""

    def __init__(self, edges, batch_size, rng):
        self.edges = edges
        self.batch_size = batch_size
        self.rng = rng
        self.order = []
        self.pos = 0

    def _reshuffle(self):
        self.order = list(range(len(self.edges)))
        self.rng.shuffle(self.order)
        self.pos = 0

    def next_batch(self):
        if self.pos + self.batch_size > len(self.order):
            self._reshuffle()
            if len(self.order) < 2:
                return []
        batch_idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return [self.edges[i] for i in batch_idx]


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


def loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets, alpha):
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
    mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
    return mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU), z_a


def measure_initial_magnitudes(model, embeddings, idx, rel_edges, tail_edges, all_positive_targets, item_ids,
                                rng, n_batches):
    model.eval()
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    tail_sampler = CyclicSampler(tail_edges, BATCH_SIZE, rng)
    rel_vals, tail_vals = [], []
    with torch.no_grad():
        for b in range(n_batches):
            rel_batch_idx = rel_order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(rel_batch_idx) < 2:
                continue
            rel_batch = [rel_edges[i] for i in rel_batch_idx]
            rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)
            rel_loss, _ = loss_for_batch(model, embeddings, idx, rel_batch, rel_negs, all_positive_targets, alpha=1.0)
            rel_vals.append(rel_loss.item())

            tail_batch = tail_sampler.next_batch()
            if len(tail_batch) < 2:
                continue
            tail_negs = build_negatives_for_edges(tail_batch, all_positive_targets, item_ids, rng, R_NEG)
            tail_loss, _ = loss_for_batch(model, embeddings, idx, tail_batch, tail_negs, all_positive_targets, alpha=0.0)
            tail_vals.append(tail_loss.item())
    return float(np.mean(rel_vals)), float(np.mean(tail_vals))


def grad_norm_check(model, embeddings, idx, rel_edges, tail_edges, all_positive_targets, item_ids, rng, weight_tail):
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    rel_batch = [rel_edges[i] for i in rel_order[:BATCH_SIZE]]
    rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)

    tail_order = list(range(len(tail_edges)))
    rng.shuffle(tail_order)
    tail_batch = [tail_edges[i] for i in tail_order[:BATCH_SIZE]]
    tail_negs = build_negatives_for_edges(tail_batch, all_positive_targets, item_ids, rng, R_NEG)

    def net_grad_norm(loss):
        model.zero_grad()
        loss.backward()
        total = 0.0
        for p in model.net.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    model.train()
    rel_loss, _ = loss_for_batch(model, embeddings, idx, rel_batch, rel_negs, all_positive_targets, alpha=1.0)
    rel_grad_norm = net_grad_norm(rel_loss)

    tail_loss, _ = loss_for_batch(model, embeddings, idx, tail_batch, tail_negs, all_positive_targets, alpha=0.0)
    tail_grad_norm_unweighted = net_grad_norm(tail_loss)

    tail_loss2, _ = loss_for_batch(model, embeddings, idx, tail_batch, tail_negs, all_positive_targets, alpha=0.0)
    tail_grad_norm_weighted = net_grad_norm(weight_tail * tail_loss2)

    model.zero_grad()
    return rel_grad_norm, tail_grad_norm_unweighted, tail_grad_norm_weighted


def run_epoch_train(model, optimizer, embeddings, idx, rel_edges, tail_edges, all_positive_targets, item_ids,
                     rng, weight_tail, uniformity_enabled=False, log_every=2000):
    model.train()
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    tail_sampler = CyclicSampler(tail_edges, BATCH_SIZE, rng)
    total_rel, total_tail, n_batches = 0.0, 0.0, 0

    for start in range(0, len(rel_order), BATCH_SIZE):
        rel_batch_idx = rel_order[start:start + BATCH_SIZE]
        if len(rel_batch_idx) < 2:
            continue
        rel_batch = [rel_edges[i] for i in rel_batch_idx]
        tail_batch = tail_sampler.next_batch()
        if len(tail_batch) < 2:
            continue

        rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)
        tail_negs = build_negatives_for_edges(tail_batch, all_positive_targets, item_ids, rng, R_NEG)

        rel_loss, z_a_rel = loss_for_batch(model, embeddings, idx, rel_batch, rel_negs, all_positive_targets, alpha=1.0)
        tail_loss, z_a_tail = loss_for_batch(model, embeddings, idx, tail_batch, tail_negs, all_positive_targets, alpha=0.0)

        loss = rel_loss + weight_tail * tail_loss
        if uniformity_enabled:
            loss = loss + UNIFORMITY_WEIGHT * uniformity_loss(z_a_rel) + UNIFORMITY_WEIGHT * uniformity_loss(z_a_tail)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_rel += rel_loss.item()
        total_tail += tail_loss.item()
        n_batches += 1
        if log_every and n_batches % log_every == 0:
            print(f"    ...batch {n_batches}/{len(rel_order)//BATCH_SIZE}, "
                  f"relevance={total_rel/n_batches:.4f} tail={total_tail/n_batches:.4f}")

    return total_rel / max(n_batches, 1), total_tail / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, idx, rel_val, tail_val, frozen_rel_val_negs, frozen_tail_val_negs,
                   all_positive_targets):
    model.eval()
    total_rel, n_rel_batches = 0.0, 0
    for start in range(0, len(rel_val), BATCH_SIZE):
        batch_edges = rel_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_rel_val_negs[start:start + BATCH_SIZE]
        loss, _ = loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets, alpha=1.0)
        total_rel += loss.item()
        n_rel_batches += 1

    total_tail, n_tail_batches = 0.0, 0
    for start in range(0, len(tail_val), BATCH_SIZE):
        batch_edges = tail_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_tail_val_negs[start:start + BATCH_SIZE]
        loss, _ = loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets, alpha=0.0)
        total_tail += loss.item()
        n_tail_batches += 1

    return total_rel / max(n_rel_batches, 1), total_tail / max(n_tail_batches, 1)

"""Shared data loading, batch construction, and loss-balancing helpers for
phase 18b. Adapted from phase 18's `train_core.py` -- the architecture,
`comp_rel`/`sub_rel`/`comp_tail` signals, and overall training procedure are
UNCHANGED. The one real difference: `sub_tail` is no longer a ranking-
distillation KL loss over anchors-only batches -- it's now a genuine MNRL
contrastive loss over its own (anchor, tail-tier-neighbor) positive pairs
(`01_build_contrastive_subtail_signal.py`), with NEGATIVES deliberately
biased toward head-tier items (the specific fix this phase tests).

THREE independent batch streams now (phase 18 had two):
- REL stream (phase 7's 76,293-edge also_buy set) -> supplies `comp_rel`
  (MNRL, anchor+target+random negatives) AND `sub_rel` (KL ranking-
  distillation, anchor only, same batch, unchanged from phase 18).
- CT stream (phase 16c's 37,922-edge attribute-pair set, reused unchanged)
  -> supplies `comp_tail` (MNRL, anchor+target+random negatives, unchanged).
- ST stream (this phase's new 74,157-edge contrastive tail-neighbor set)
  -> supplies `sub_tail` (MNRL, anchor+target+HEAD-BIASED negatives -- the
  new part).

One epoch = one full pass over the largest population (REL, 68,664 train
edges); CT and ST are both cycled (34,130 and 66,742 respectively).
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import FourHeadDedicatedCapacity, kl_ranking_distillation_loss, mean_pairwise_cosine, mnrl_loss

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
PHASE16C_DIR = BASE_DIR.parent / "phase16c_attribute_tail_signal"
PHASE18_DIR = BASE_DIR.parent / "phase18_unified_two_axis"
TIER_LOOKUP_CSV = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"

EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"
REL_EDGES_JSON = PHASE7_DIR / "data" / "positive_edges.json"  # relevance-axis stream (also_buy)
CT_EDGES_JSON = PHASE16C_DIR / "data" / "attribute_pairs.json"  # comp_tail stream, reused unchanged
ST_EDGES_JSON = BASE_DIR / "data" / "contrastive_subtail_pairs.json"  # NEW: sub_tail contrastive stream
UNRESTRICTED_NN_NPZ = PHASE18_DIR / "data" / "unrestricted_nn_lookup.npz"  # sub_rel teacher, reused unchanged

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU = 0.07
R_NEG = 8
HEAD_BIASED_FRACTION = 0.5  # half of sub_tail's negatives drawn specifically from head-tier items
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-4
N_CALIBRATION_BATCHES = 20
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CORNERS = ("sub_rel", "sub_tail", "comp_rel", "comp_tail")


class Edge:
    __slots__ = ("source", "target")

    def __init__(self, source, target):
        self.source = source
        self.target = target


def _load_edges(path):
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

    rel_train, rel_val = _load_edges(REL_EDGES_JSON)
    ct_train, ct_val = _load_edges(CT_EDGES_JSON)
    st_train, st_val = _load_edges(ST_EDGES_JSON)

    all_positive_targets = {}
    for e in rel_train + rel_val + ct_train + ct_val + st_train + st_val:
        all_positive_targets.setdefault(e.source, set()).add(e.target)

    nn_unrestricted = np.load(UNRESTRICTED_NN_NPZ, allow_pickle=True)
    nn_unrestricted_ids = [str(a) for a in nn_unrestricted["item_ids"]]
    assert nn_unrestricted_ids == item_ids, "unrestricted NN lookup ordering must match embedding ordering"
    sr_nn_indices = nn_unrestricted["indices"]
    sr_nn_sims = nn_unrestricted["sims"].astype(np.float32)

    tier_df = pd.read_csv(TIER_LOOKUP_CSV, usecols=["asin", "tier"])
    tier_df["asin"] = tier_df["asin"].astype(str)
    tier_lookup = dict(zip(tier_df["asin"], tier_df["tier"]))
    head_tier_ids = [a for a in item_ids if tier_lookup.get(a) == "head"]

    return (embeddings, idx, item_ids, rel_train, rel_val, ct_train, ct_val, st_train, st_val,
            all_positive_targets, sr_nn_indices, sr_nn_sims, head_tier_ids)


class CyclicSampler:
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


def sample_head_biased_negatives(rng, anchor, positive_set, head_tier_ids, item_ids, k, head_fraction):
    """THE FIX: half (by default) of sub_tail's negatives are drawn
    specifically from head-tier items -- the exact population the loss needs
    to learn to push away from, per this phase's diagnosis. The rest are
    uniform-random, matching comp_tail's general-purpose negative sampling
    for robustness."""
    n_head = round(k * head_fraction)
    n_random = k - n_head
    exclude = positive_set | {anchor}

    negs = []
    n_head_pool = len(head_tier_ids)
    tries, max_tries = 0, n_head * 20
    while len(negs) < n_head and tries < max_tries:
        cand = head_tier_ids[rng.randrange(n_head_pool)]
        if cand not in exclude and cand not in negs:
            negs.append(cand)
        tries += 1
    while len(negs) < n_head:  # pool is large (9,961) relative to n_head=4; this should essentially never fire
        negs.append(head_tier_ids[rng.randrange(n_head_pool)])

    negs.extend(sample_random_negatives(rng, anchor, positive_set | set(negs), len(item_ids), item_ids, n_random))
    return negs


def build_head_biased_negatives_for_edges(batch_edges, all_positive_targets, item_ids, head_tier_ids, rng, k,
                                           head_fraction=HEAD_BIASED_FRACTION):
    out = []
    for e in batch_edges:
        positive_set = all_positive_targets.get(e.source, set())
        out.append(sample_head_biased_negatives(rng, e.source, positive_set, head_tier_ids, item_ids, k,
                                                  head_fraction))
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


def mnrl_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets, corner):
    """Generic MNRL loss for any corner using explicit (anchor, target,
    negatives) triples -- used for comp_rel, comp_tail, AND (new this
    phase) sub_tail."""
    anchors = [e.source for e in batch_edges]
    positives = [e.target for e in batch_edges]
    a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
    p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
    extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
    extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

    z_a = model.single_head_output(a_emb, corner)
    z_p = model.single_head_output(p_emb, corner)
    B, K, D = extra_emb.shape
    z_extra = model.single_head_output(extra_emb.reshape(B * K, D), corner).reshape(B, K, -1)
    mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
    return mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU), z_a


def sub_rel_loss_for_batch(model, embeddings, idx, batch_edges, nn_indices, nn_sims):
    """KL ranking-distillation loss for sub_rel -- UNCHANGED from phase 18.
    Anchors are the same REL-stream batch's sources (targets unused)."""
    anchors = [e.source for e in batch_edges]
    anchor_gidx = np.array([idx[a] for a in anchors])
    neighbor_gidx = nn_indices[anchor_gidx]
    B, K = neighbor_gidx.shape

    teacher_sims = torch.tensor(nn_sims[anchor_gidx], device=DEVICE)
    anchor_emb = torch.tensor(embeddings[anchor_gidx], device=DEVICE)
    neighbor_emb = torch.tensor(embeddings[neighbor_gidx.reshape(-1)], device=DEVICE)

    z_anchor = model.single_head_output(anchor_emb, "sub_rel")
    z_neighbors = model.single_head_output(neighbor_emb, "sub_rel").reshape(B, K, -1)
    return kl_ranking_distillation_loss(z_anchor, z_neighbors, teacher_sims, tau=TAU), z_anchor


def all_four_losses(model, embeddings, idx, rel_batch, ct_batch, st_batch, rel_negs, ct_negs, st_negs,
                     all_positive_targets, sr_nn_indices, sr_nn_sims):
    comp_rel_loss, z_comp_rel = mnrl_loss_for_batch(model, embeddings, idx, rel_batch, rel_negs,
                                                      all_positive_targets, "comp_rel")
    sub_rel_loss, z_sub_rel = sub_rel_loss_for_batch(model, embeddings, idx, rel_batch, sr_nn_indices, sr_nn_sims)
    comp_tail_loss, z_comp_tail = mnrl_loss_for_batch(model, embeddings, idx, ct_batch, ct_negs,
                                                        all_positive_targets, "comp_tail")
    sub_tail_loss, z_sub_tail = mnrl_loss_for_batch(model, embeddings, idx, st_batch, st_negs,
                                                      all_positive_targets, "sub_tail")
    losses = {"comp_rel": comp_rel_loss, "sub_rel": sub_rel_loss, "comp_tail": comp_tail_loss,
              "sub_tail": sub_tail_loss}
    collapse_samples = {"comp_rel": z_comp_rel, "sub_rel": z_sub_rel, "comp_tail": z_comp_tail,
                         "sub_tail": z_sub_tail}
    return losses, collapse_samples


def measure_initial_magnitudes(model, embeddings, idx, rel_edges, ct_edges, st_edges, all_positive_targets,
                                item_ids, head_tier_ids, sr_nn_indices, sr_nn_sims, rng, n_batches):
    model.eval()
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    ct_sampler = CyclicSampler(ct_edges, BATCH_SIZE, rng)
    st_sampler = CyclicSampler(st_edges, BATCH_SIZE, rng)
    vals = {c: [] for c in CORNERS}
    with torch.no_grad():
        for b in range(n_batches):
            rel_batch_idx = rel_order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(rel_batch_idx) < 2:
                continue
            rel_batch = [rel_edges[i] for i in rel_batch_idx]
            ct_batch = ct_sampler.next_batch()
            st_batch = st_sampler.next_batch()
            if len(ct_batch) < 2 or len(st_batch) < 2:
                continue
            rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)
            ct_negs = build_negatives_for_edges(ct_batch, all_positive_targets, item_ids, rng, R_NEG)
            st_negs = build_head_biased_negatives_for_edges(st_batch, all_positive_targets, item_ids,
                                                              head_tier_ids, rng, R_NEG)
            losses, _ = all_four_losses(model, embeddings, idx, rel_batch, ct_batch, st_batch, rel_negs, ct_negs,
                                         st_negs, all_positive_targets, sr_nn_indices, sr_nn_sims)
            for c in CORNERS:
                vals[c].append(losses[c].item())
    return {c: float(np.mean(vals[c])) for c in CORNERS}


def grad_norm_check(model, embeddings, idx, rel_edges, ct_edges, st_edges, all_positive_targets, item_ids,
                     head_tier_ids, sr_nn_indices, sr_nn_sims, rng, weights):
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    rel_batch = [rel_edges[i] for i in rel_order[:BATCH_SIZE]]
    rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)

    ct_order = list(range(len(ct_edges)))
    rng.shuffle(ct_order)
    ct_batch = [ct_edges[i] for i in ct_order[:BATCH_SIZE]]
    ct_negs = build_negatives_for_edges(ct_batch, all_positive_targets, item_ids, rng, R_NEG)

    st_order = list(range(len(st_edges)))
    rng.shuffle(st_order)
    st_batch = [st_edges[i] for i in st_order[:BATCH_SIZE]]
    st_negs = build_head_biased_negatives_for_edges(st_batch, all_positive_targets, item_ids, head_tier_ids, rng,
                                                      R_NEG)

    def shared_grad_norm(loss):
        model.zero_grad()
        loss.backward()
        total = 0.0
        for p in model.shared.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5

    model.train()
    results = {}
    loss, _ = mnrl_loss_for_batch(model, embeddings, idx, rel_batch, rel_negs, all_positive_targets, "comp_rel")
    results["comp_rel_unweighted"] = shared_grad_norm(loss)
    loss, _ = mnrl_loss_for_batch(model, embeddings, idx, ct_batch, ct_negs, all_positive_targets, "comp_tail")
    results["comp_tail_unweighted"] = shared_grad_norm(loss)
    loss, _ = mnrl_loss_for_batch(model, embeddings, idx, st_batch, st_negs, all_positive_targets, "sub_tail")
    results["sub_tail_unweighted"] = shared_grad_norm(loss)
    loss, _ = sub_rel_loss_for_batch(model, embeddings, idx, rel_batch, sr_nn_indices, sr_nn_sims)
    results["sub_rel_unweighted"] = shared_grad_norm(loss)

    loss, _ = mnrl_loss_for_batch(model, embeddings, idx, ct_batch, ct_negs, all_positive_targets, "comp_tail")
    results["comp_tail_weighted"] = shared_grad_norm(weights["comp_tail"] * loss)
    loss, _ = mnrl_loss_for_batch(model, embeddings, idx, st_batch, st_negs, all_positive_targets, "sub_tail")
    results["sub_tail_weighted"] = shared_grad_norm(weights["sub_tail"] * loss)
    loss, _ = sub_rel_loss_for_batch(model, embeddings, idx, rel_batch, sr_nn_indices, sr_nn_sims)
    results["sub_rel_weighted"] = shared_grad_norm(weights["sub_rel"] * loss)

    model.zero_grad()
    return results


def run_epoch_train(model, optimizer, embeddings, idx, rel_edges, ct_edges, st_edges, all_positive_targets,
                     item_ids, head_tier_ids, sr_nn_indices, sr_nn_sims, rng, weights, log_every=2000):
    model.train()
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    ct_sampler = CyclicSampler(ct_edges, BATCH_SIZE, rng)
    st_sampler = CyclicSampler(st_edges, BATCH_SIZE, rng)
    totals = {c: 0.0 for c in CORNERS}
    n_batches = 0

    for start in range(0, len(rel_order), BATCH_SIZE):
        rel_batch_idx = rel_order[start:start + BATCH_SIZE]
        if len(rel_batch_idx) < 2:
            continue
        rel_batch = [rel_edges[i] for i in rel_batch_idx]
        ct_batch = ct_sampler.next_batch()
        st_batch = st_sampler.next_batch()
        if len(ct_batch) < 2 or len(st_batch) < 2:
            continue

        rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)
        ct_negs = build_negatives_for_edges(ct_batch, all_positive_targets, item_ids, rng, R_NEG)
        st_negs = build_head_biased_negatives_for_edges(st_batch, all_positive_targets, item_ids, head_tier_ids,
                                                          rng, R_NEG)

        losses, _ = all_four_losses(model, embeddings, idx, rel_batch, ct_batch, st_batch, rel_negs, ct_negs,
                                     st_negs, all_positive_targets, sr_nn_indices, sr_nn_sims)
        loss = sum(weights[c] * losses[c] for c in CORNERS)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for c in CORNERS:
            totals[c] += losses[c].item()
        n_batches += 1
        if log_every and n_batches % log_every == 0:
            summary = " ".join(f"{c}={totals[c]/n_batches:.4f}" for c in CORNERS)
            print(f"    ...batch {n_batches}/{len(rel_order)//BATCH_SIZE}, {summary}")

    return {c: totals[c] / max(n_batches, 1) for c in CORNERS}


@torch.no_grad()
def run_epoch_val(model, embeddings, idx, rel_val, ct_val, st_val, frozen_rel_val_negs, frozen_ct_val_negs,
                   frozen_st_val_negs, all_positive_targets, sr_nn_indices, sr_nn_sims):
    model.eval()
    totals = {c: 0.0 for c in CORNERS}
    n_rel_batches, n_ct_batches, n_st_batches = 0, 0, 0

    for start in range(0, len(rel_val), BATCH_SIZE):
        batch_edges = rel_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_rel_val_negs[start:start + BATCH_SIZE]
        comp_loss, _ = mnrl_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets,
                                            "comp_rel")
        sub_loss, _ = sub_rel_loss_for_batch(model, embeddings, idx, batch_edges, sr_nn_indices, sr_nn_sims)
        totals["comp_rel"] += comp_loss.item()
        totals["sub_rel"] += sub_loss.item()
        n_rel_batches += 1

    for start in range(0, len(ct_val), BATCH_SIZE):
        batch_edges = ct_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_ct_val_negs[start:start + BATCH_SIZE]
        loss, _ = mnrl_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets,
                                       "comp_tail")
        totals["comp_tail"] += loss.item()
        n_ct_batches += 1

    for start in range(0, len(st_val), BATCH_SIZE):
        batch_edges = st_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_st_val_negs[start:start + BATCH_SIZE]
        loss, _ = mnrl_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets,
                                       "sub_tail")
        totals["sub_tail"] += loss.item()
        n_st_batches += 1

    return {
        "comp_rel": totals["comp_rel"] / max(n_rel_batches, 1),
        "sub_rel": totals["sub_rel"] / max(n_rel_batches, 1),
        "comp_tail": totals["comp_tail"] / max(n_ct_batches, 1),
        "sub_tail": totals["sub_tail"] / max(n_st_batches, 1),
    }

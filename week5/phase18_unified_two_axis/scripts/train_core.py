"""Shared data loading, batch construction, and loss-balancing helpers for
phase 18, imported by 03_smoke_test.py and 04_train.py.

Four corners, two independent batch STREAMS (not four) -- mirroring phase
12c/17's own precedent of reusing one edge batch for both a complement loss
(needs anchor+target) and a substitute loss (only needs the anchor), applied
here to each axis position separately:

- The RELEVANCE-axis stream (phase 7's full 76,293-edge also_buy set, same
  data phase 16/16d's relevance signal used) supplies anchors+targets for
  `comp_rel`'s MNRL loss, and the SAME batch's anchors for `sub_rel`'s KL
  ranking-distillation loss (teacher: step 2's unrestricted NN lookup).
- The TAIL-EXPOSURE-axis stream (phase 16c's 37,922-edge attribute-pair
  set, REUSED DIRECTLY, same data phase 16d's tail signal used) supplies
  anchors+targets for `comp_tail`'s MNRL loss, and the SAME batch's anchors
  for `sub_tail`'s KL ranking-distillation loss (teacher: step 1's new
  tail-restricted NN lookup).

One epoch = one full pass over the larger (relevance) population; the
smaller (tail) population is cycled, same convention as phases 16/16c/16d.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import FourHeadDedicatedCapacity, kl_ranking_distillation_loss, mean_pairwise_cosine, mnrl_loss

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
PHASE16C_DIR = BASE_DIR.parent / "phase16c_attribute_tail_signal"

EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"
REL_EDGES_JSON = PHASE7_DIR / "data" / "positive_edges.json"  # relevance-axis stream (also_buy)
TAIL_EDGES_JSON = PHASE16C_DIR / "data" / "attribute_pairs.json"  # tail-exposure-axis stream, reused directly
UNRESTRICTED_NN_NPZ = BASE_DIR / "data" / "unrestricted_nn_lookup.npz"  # sub_rel teacher
TAIL_NN_NPZ = BASE_DIR / "data" / "tail_nn_lookup.npz"  # sub_tail teacher

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
    tail_train, tail_val = _load_edges(TAIL_EDGES_JSON)

    all_positive_targets = {}
    for e in rel_train + rel_val + tail_train + tail_val:
        all_positive_targets.setdefault(e.source, set()).add(e.target)

    nn_unrestricted = np.load(UNRESTRICTED_NN_NPZ, allow_pickle=True)
    nn_unrestricted_ids = [str(a) for a in nn_unrestricted["item_ids"]]
    assert nn_unrestricted_ids == item_ids, "unrestricted NN lookup ordering must match embedding ordering"
    sr_nn_indices = nn_unrestricted["indices"]
    sr_nn_sims = nn_unrestricted["sims"].astype(np.float32)

    nn_tail = np.load(TAIL_NN_NPZ, allow_pickle=True)
    nn_tail_ids = [str(a) for a in nn_tail["item_ids"]]
    assert nn_tail_ids == item_ids, "tail-restricted NN lookup ordering must match embedding ordering"
    st_nn_indices = nn_tail["indices"]
    st_nn_sims = nn_tail["sims"].astype(np.float32)

    return (embeddings, idx, item_ids, rel_train, rel_val, tail_train, tail_val, all_positive_targets,
            sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims)


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


def build_inbatch_mask(batch_edges, all_positive_targets):
    B = len(batch_edges)
    mask = np.zeros((B, B), dtype=bool)
    for i, ei in enumerate(batch_edges):
        positives_i = all_positive_targets.get(ei.source, set())
        for j, ej in enumerate(batch_edges):
            if i != j and ej.target in positives_i:
                mask[i, j] = True
    return mask


def comp_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets, corner):
    """MNRL loss for a complement corner (comp_rel or comp_tail)."""
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


def sub_loss_for_batch(model, embeddings, idx, batch_edges, nn_indices, nn_sims, corner):
    """KL ranking-distillation loss for a substitute corner (sub_rel or
    sub_tail) -- anchors are the SAME batch's sources (targets unused, per
    phase 12c/17 precedent)."""
    anchors = [e.source for e in batch_edges]
    anchor_gidx = np.array([idx[a] for a in anchors])
    neighbor_gidx = nn_indices[anchor_gidx]
    B, K = neighbor_gidx.shape

    teacher_sims = torch.tensor(nn_sims[anchor_gidx], device=DEVICE)
    anchor_emb = torch.tensor(embeddings[anchor_gidx], device=DEVICE)
    neighbor_emb = torch.tensor(embeddings[neighbor_gidx.reshape(-1)], device=DEVICE)

    z_anchor = model.single_head_output(anchor_emb, corner)
    z_neighbors = model.single_head_output(neighbor_emb, corner).reshape(B, K, -1)

    return kl_ranking_distillation_loss(z_anchor, z_neighbors, teacher_sims, tau=TAU), z_anchor


def all_four_losses(model, embeddings, idx, rel_batch, tail_batch, rel_negs, tail_negs, all_positive_targets,
                     sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims):
    comp_rel_loss, z_comp_rel = comp_loss_for_batch(model, embeddings, idx, rel_batch, rel_negs,
                                                      all_positive_targets, "comp_rel")
    sub_rel_loss, z_sub_rel = sub_loss_for_batch(model, embeddings, idx, rel_batch, sr_nn_indices, sr_nn_sims,
                                                  "sub_rel")
    comp_tail_loss, z_comp_tail = comp_loss_for_batch(model, embeddings, idx, tail_batch, tail_negs,
                                                        all_positive_targets, "comp_tail")
    sub_tail_loss, z_sub_tail = sub_loss_for_batch(model, embeddings, idx, tail_batch, st_nn_indices, st_nn_sims,
                                                    "sub_tail")
    losses = {"comp_rel": comp_rel_loss, "sub_rel": sub_rel_loss, "comp_tail": comp_tail_loss,
              "sub_tail": sub_tail_loss}
    collapse_samples = {"comp_rel": z_comp_rel, "sub_rel": z_sub_rel, "comp_tail": z_comp_tail,
                         "sub_tail": z_sub_tail}
    return losses, collapse_samples


def measure_initial_magnitudes(model, embeddings, idx, rel_edges, tail_edges, all_positive_targets, item_ids,
                                sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims, rng, n_batches):
    model.eval()
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    tail_sampler = CyclicSampler(tail_edges, BATCH_SIZE, rng)
    vals = {c: [] for c in CORNERS}
    with torch.no_grad():
        for b in range(n_batches):
            rel_batch_idx = rel_order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(rel_batch_idx) < 2:
                continue
            rel_batch = [rel_edges[i] for i in rel_batch_idx]
            tail_batch = tail_sampler.next_batch()
            if len(tail_batch) < 2:
                continue
            rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)
            tail_negs = build_negatives_for_edges(tail_batch, all_positive_targets, item_ids, rng, R_NEG)
            losses, _ = all_four_losses(model, embeddings, idx, rel_batch, tail_batch, rel_negs, tail_negs,
                                         all_positive_targets, sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims)
            for c in CORNERS:
                vals[c].append(losses[c].item())
    return {c: float(np.mean(vals[c])) for c in CORNERS}


def grad_norm_check(model, embeddings, idx, rel_edges, tail_edges, all_positive_targets, item_ids,
                     sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims, rng, weights):
    """Gradient L2-norm into `model.shared.parameters()` -- the only shared
    submodule -- for each of the 4 corners individually, isolated, plus each
    non-reference corner after weighting."""
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    rel_batch = [rel_edges[i] for i in rel_order[:BATCH_SIZE]]
    rel_negs = build_negatives_for_edges(rel_batch, all_positive_targets, item_ids, rng, R_NEG)

    tail_order = list(range(len(tail_edges)))
    rng.shuffle(tail_order)
    tail_batch = [tail_edges[i] for i in tail_order[:BATCH_SIZE]]
    tail_negs = build_negatives_for_edges(tail_batch, all_positive_targets, item_ids, rng, R_NEG)

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
    for corner, batch, negs in [("comp_rel", rel_batch, rel_negs), ("comp_tail", tail_batch, tail_negs)]:
        loss, _ = comp_loss_for_batch(model, embeddings, idx, batch, negs, all_positive_targets, corner)
        results[f"{corner}_unweighted"] = shared_grad_norm(loss)
    for corner, batch, nn_idx, nn_sim in [("sub_rel", rel_batch, sr_nn_indices, sr_nn_sims),
                                           ("sub_tail", tail_batch, st_nn_indices, st_nn_sims)]:
        loss, _ = sub_loss_for_batch(model, embeddings, idx, batch, nn_idx, nn_sim, corner)
        results[f"{corner}_unweighted"] = shared_grad_norm(loss)

    # weighted versions (recomputed fresh, since the graph from above was already consumed by backward)
    for corner, batch, negs in [("comp_tail", tail_batch, tail_negs)]:
        loss, _ = comp_loss_for_batch(model, embeddings, idx, batch, negs, all_positive_targets, corner)
        results[f"{corner}_weighted"] = shared_grad_norm(weights[corner] * loss)
    for corner, batch, nn_idx, nn_sim in [("sub_rel", rel_batch, sr_nn_indices, sr_nn_sims),
                                           ("sub_tail", tail_batch, st_nn_indices, st_nn_sims)]:
        loss, _ = sub_loss_for_batch(model, embeddings, idx, batch, nn_idx, nn_sim, corner)
        results[f"{corner}_weighted"] = shared_grad_norm(weights[corner] * loss)

    model.zero_grad()
    return results


def run_epoch_train(model, optimizer, embeddings, idx, rel_edges, tail_edges, all_positive_targets, item_ids,
                     sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims, rng, weights, log_every=2000):
    model.train()
    rel_order = list(range(len(rel_edges)))
    rng.shuffle(rel_order)
    tail_sampler = CyclicSampler(tail_edges, BATCH_SIZE, rng)
    totals = {c: 0.0 for c in CORNERS}
    n_batches = 0

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

        losses, _ = all_four_losses(model, embeddings, idx, rel_batch, tail_batch, rel_negs, tail_negs,
                                     all_positive_targets, sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims)
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
def run_epoch_val(model, embeddings, idx, rel_val, tail_val, frozen_rel_val_negs, frozen_tail_val_negs,
                   all_positive_targets, sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims):
    model.eval()
    totals = {c: 0.0 for c in CORNERS}
    n_rel_batches, n_tail_batches = 0, 0

    for start in range(0, len(rel_val), BATCH_SIZE):
        batch_edges = rel_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_rel_val_negs[start:start + BATCH_SIZE]
        comp_loss, _ = comp_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets,
                                            "comp_rel")
        sub_loss, _ = sub_loss_for_batch(model, embeddings, idx, batch_edges, sr_nn_indices, sr_nn_sims, "sub_rel")
        totals["comp_rel"] += comp_loss.item()
        totals["sub_rel"] += sub_loss.item()
        n_rel_batches += 1

    for start in range(0, len(tail_val), BATCH_SIZE):
        batch_edges = tail_val[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_tail_val_negs[start:start + BATCH_SIZE]
        comp_loss, _ = comp_loss_for_batch(model, embeddings, idx, batch_edges, neg_lists, all_positive_targets,
                                            "comp_tail")
        sub_loss, _ = sub_loss_for_batch(model, embeddings, idx, batch_edges, st_nn_indices, st_nn_sims, "sub_tail")
        totals["comp_tail"] += comp_loss.item()
        totals["sub_tail"] += sub_loss.item()
        n_tail_batches += 1

    return {
        "comp_rel": totals["comp_rel"] / max(n_rel_batches, 1),
        "sub_rel": totals["sub_rel"] / max(n_rel_batches, 1),
        "comp_tail": totals["comp_tail"] / max(n_tail_batches, 1),
        "sub_tail": totals["sub_tail"] / max(n_tail_batches, 1),
    }

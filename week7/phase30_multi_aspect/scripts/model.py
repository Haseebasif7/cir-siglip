"""
Phase 30 shared model code: MultiAspectProjectionHead, replacing phase 27/28's
single-vector ProjectionHeadGeneral with K=4 separately-normalized 32-d aspect
vectors plus a learned, query-conditioned aggregation layer, per the brief's
exact formula.

Shared trunk (1536 -> 1024, ReLU, Dropout) is identical in shape to phase 27's
ProjectionHeadGeneral's own hidden layer -- only what happens after the hidden
layer changes (four small heads instead of one 1024->128 head).

Kept as a standalone file (not imported from phase 27), per this project's
established cross-phase convention of each phase folder being self-contained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ASPECT_NAMES = ["visual", "form", "semantic", "general"]
N_ASPECTS = len(ASPECT_NAMES)
ASPECT_DIM = 32  # K x ASPECT_DIM = 128, matching phase 27/28's total output dim


class MultiAspectProjectionHead(nn.Module):
    """Produces K=4 independently L2-normalized 32-d aspect vectors per item,
    plus (for use in query role) a learned softmax aggregation-weight layer
    over the query's own concatenated aspect vectors.

    forward(x) -> (B, K, ASPECT_DIM) aspect vectors, each row L2-normalized.
    agg_weights(aspect_vecs) -> (B, K) softmax weights, computed from the
    concatenated (B, K*ASPECT_DIM) aspect vectors -- query-side only, per the
    brief ("computed from the query itself"). Never conditioned on the
    candidate, unlike phase 29's candidate-conditioned cross-attention.
    """

    def __init__(self, in_dim=1536, hidden_dim=1024, n_aspects=N_ASPECTS,
                 aspect_dim=ASPECT_DIM, dropout=0.1):
        super().__init__()
        self.n_aspects = n_aspects
        self.aspect_dim = aspect_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        # Four independent small heads branching off the shared trunk, named
        # per the brief's stated targets (names are documentation of intent,
        # not enforced -- nothing in the loss ties a specific head to a
        # specific semantic meaning, that's left for the model to discover).
        self.aspect_heads = nn.ModuleList([
            nn.Linear(hidden_dim, aspect_dim) for _ in range(n_aspects)
        ])
        # Aggregation weight layer: concatenated K*aspect_dim query vector -> K logits -> softmax.
        self.agg_layer = nn.Linear(n_aspects * aspect_dim, n_aspects)

    def forward(self, x):
        h = self.trunk(x)
        aspects = [F.normalize(head(h), p=2, dim=-1) for head in self.aspect_heads]
        return torch.stack(aspects, dim=-2)  # (..., n_aspects, aspect_dim)

    def agg_weights(self, aspect_vecs):
        """aspect_vecs: (..., n_aspects, aspect_dim) -> (..., n_aspects) softmax weights."""
        flat = aspect_vecs.flatten(start_dim=-2)  # (..., n_aspects * aspect_dim)
        return F.softmax(self.agg_layer(flat), dim=-1)


def aspect_score(query_aspects, query_weights, candidate_aspects):
    """query_aspects: (B, K, D). query_weights: (B, K). candidate_aspects: (B, K, D)
    (aligned, one candidate per query -- used for the positive/anchor role) OR
    (B, R, K, D) (many candidates per query -- used for extra negatives) OR
    a plain (N, K, D) pool shared across all queries (in-batch / eval pool).
    Returns per-query-candidate weighted-sum-of-per-aspect-cosine scores.
    """
    if candidate_aspects.dim() == query_aspects.dim():
        # one candidate per query, aligned: (B,K,D) x (B,K,D) -> (B,K) -> (B,)
        sim_k = (query_aspects * candidate_aspects).sum(dim=-1)  # (B, K)
        return (query_weights * sim_k).sum(dim=-1)  # (B,)
    elif candidate_aspects.dim() == query_aspects.dim() + 1:
        # (B, R, K, D) extra negatives per query
        sim_k = torch.einsum("bkd,brkd->brk", query_aspects, candidate_aspects)  # (B, R, K)
        return torch.einsum("brk,bk->br", sim_k, query_weights)  # (B, R)
    else:
        raise ValueError(f"unexpected candidate_aspects shape {candidate_aspects.shape}")


def inbatch_scores(query_aspects, query_weights, pool_aspects):
    """query_aspects: (B,K,D), query_weights: (B,K), pool_aspects: (P,K,D)
    (P may equal B for in-batch positives, or be a full candidate pool at eval
    time). Returns (B, P) weighted-sum-of-per-aspect-cosine score matrix."""
    sim_k = torch.einsum("bkd,pkd->bpk", query_aspects, pool_aspects)  # (B, P, K)
    return torch.einsum("bpk,bk->bp", sim_k, query_weights)  # (B, P)


def mnrl_loss_multiaspect(anchor_aspects, anchor_weights, positive_aspects,
                           extra_aspects, mask, tau):
    """Multi-aspect analogue of phase 9/23/25/26/27's mnrl_loss. The anchor
    plays the query role (aggregation weights come from it only, per the
    brief); positive and extra-negative candidates are scored against it via
    the weighted per-aspect cosine sum instead of a single dot product."""
    inbatch = inbatch_scores(anchor_aspects, anchor_weights, positive_aspects) / tau  # (B,B)
    inbatch = inbatch.masked_fill(mask, float("-inf"))
    extra = aspect_score(anchor_aspects, anchor_weights, extra_aspects) / tau  # (B,R)
    logits = torch.cat([inbatch, extra], dim=1)
    labels = torch.arange(anchor_aspects.shape[0], device=anchor_aspects.device)
    return F.cross_entropy(logits, labels)


def weight_entropy(weights, eps=1e-9):
    """weights: (..., K) softmax aggregation weights. Returns (raw_entropy,
    normalized_entropy) tensors, same convention as phase 29's attention
    entropy tracking: normalized = raw / log(K), 0 = one-hot collapse,
    1 = uniform (aspect bias fully disengaged, equivalent to plain averaging)."""
    K = weights.shape[-1]
    raw = -(weights * (weights + eps).log()).sum(dim=-1)
    normalized = raw / torch.log(torch.tensor(float(K), device=weights.device))
    return raw, normalized

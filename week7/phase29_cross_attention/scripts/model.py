"""
Phase 29: shared model definitions, importable both locally (for the
qualitative/attention-inspection script) and, as source text embedded
directly in modal_app.py (Modal functions need self-contained code, so
modal_app.py keeps its own copy of these classes -- see the note there).
Kept here too as the single source of truth for local scripts.

CrossAttentionScorer implements the brief's exact formula:
  attention_weights = softmax(candidate_key(C) . context_query(X_i) / sqrt(d))
  query = sum_i attention_weights[i] * context_value(X_i)
  score = query . candidate_value(C)

Four new 128->128 linear layers on top of the existing (unchanged)
1536->1024->128 ProjectionHeadGeneral. No depth, no multi-head, no
residuals/LayerNorm, per the brief's "minimum viable inductive bias"
instruction.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeadGeneral(nn.Module):
    """Unchanged from phase 27/28: 1536 -> hidden_dims -> out_dim, L2-normalized output."""

    def __init__(self, in_dim, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class CrossAttentionScorer(nn.Module):
    """The four new small linear layers, operating on the projection head's
    128-d output space. Kept as a separate module (not fused into the head)
    so proj_head's 128-d output stays reusable for both roles (context item
    and candidate) without recomputing it twice."""

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.candidate_key = nn.Linear(dim, dim)
        self.context_query = nn.Linear(dim, dim)
        self.context_value = nn.Linear(dim, dim)
        self.candidate_value = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def query_for_candidate(self, proj_context, proj_candidate, context_mask=None):
        """proj_context: (..., L, dim) projected context items (padded).
        proj_candidate: (..., dim) the single candidate C conditioning the
        attention. context_mask: (..., L) bool, True where PADDING (to be
        masked out with -inf before softmax). Returns (..., dim) L2-normalized
        context-aware query vector, conditioned on this specific candidate."""
        k_c = self.candidate_key(proj_candidate)                     # (..., dim)
        q_x = self.context_query(proj_context)                        # (..., L, dim)
        logits = torch.einsum("...ld,...d->...l", q_x, k_c) / self.scale  # (..., L)
        if context_mask is not None:
            logits = logits.masked_fill(context_mask, float("-inf"))
        weights = F.softmax(logits, dim=-1)                            # (..., L)
        v_x = self.context_value(proj_context)                         # (..., L, dim)
        query = torch.einsum("...l,...ld->...d", weights, v_x)         # (..., dim)
        return F.normalize(query, p=2, dim=-1), weights

    def value_for_candidate(self, proj_candidate):
        return F.normalize(self.candidate_value(proj_candidate), p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, mask, tau):
    """Unchanged from phase 27/28 -- reused byte-for-byte. z_a/z_p: (B, dim).
    z_extra: (B, R, dim). mask: (B, B) bool, True = mask out (false negative)."""
    B = z_a.shape[0]
    inbatch_logits = (z_a @ z_p.T) / tau
    inbatch_logits = inbatch_logits.masked_fill(mask, float("-inf"))
    extra_logits = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
    logits = torch.cat([inbatch_logits, extra_logits], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)

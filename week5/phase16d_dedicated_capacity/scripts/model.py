"""Phase 16d model code. Direct architectural fix for the capacity-
competition problem diagnosed in phase 16c: instead of one shared
768->256->128 MLP with two small additive 128-d mode vectors on top (14-16%
of the shared trunk's norm, per phase 16's own mode-vector-magnitude check),
this keeps only a MINIMAL shared first layer (768->256, dimensionality
reduction only) and gives each mode a full, independently-trained
256->128 head -- substantial, dedicated capacity, not a small correction.

`mnrl_loss` is unchanged from phases 9/12c/16/16c. `uniformity_loss` is
ported in again as an optional, off-by-default safeguard.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DedicatedCapacityHead(nn.Module):
    """Shared: Linear(768,256)->ReLU->Dropout (dimensionality reduction
    only, not the full transformation). Two fully independent, dedicated
    Linear(256,128) heads from there, each L2-normalized on its own before
    blending. At inference/training, alpha blends the two heads' own
    (already-normalized) outputs, then the blend is renormalized -- alpha=1.0
    returns exactly the relevance head's own output (tail head's gradient
    contribution is then exactly zero for that forward pass, since it's
    multiplied by (1-alpha)=0), alpha=0.0 returns exactly the tail head's.
    """

    def __init__(self, in_dim=768, shared_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.relevance_head = nn.Linear(shared_dim, out_dim)
        self.tail_head = nn.Linear(shared_dim, out_dim)

    def forward(self, x, alpha):
        h = self.shared(x)
        z_rel = F.normalize(self.relevance_head(h), p=2, dim=-1)
        z_tail = F.normalize(self.tail_head(h), p=2, dim=-1)
        if isinstance(alpha, torch.Tensor):
            blend = alpha.unsqueeze(-1) * z_rel + (1 - alpha).unsqueeze(-1) * z_tail
        else:
            blend = alpha * z_rel + (1 - alpha) * z_tail
        return F.normalize(blend, p=2, dim=-1)

    def head_outputs(self, x):
        """Both heads' own normalized outputs, unblended -- used by the
        head-similarity diagnostic (the adapted equivalent of phase
        16/16c's mode-vector cosine similarity check)."""
        h = self.shared(x)
        z_rel = F.normalize(self.relevance_head(h), p=2, dim=-1)
        z_tail = F.normalize(self.tail_head(h), p=2, dim=-1)
        return z_rel, z_tail


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Unchanged from phases 9/12c/16/16c."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, unchanged."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()


def uniformity_loss(f, t=2.0):
    """Ported again, OFF by default -- MNRL/softmax losses don't have the
    margin-loss collapse failure mode this was built for."""
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

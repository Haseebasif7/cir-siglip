"""
Phase 10 model + loss code. ProjectionHead architecture, mnrl_loss, and
mean_pairwise_cosine are copied unchanged from phase 9 (same 768->256->128
frozen-SigLIP MLP, same MNRL compatibility loss) -- this phase adds exactly
one new piece, invariance_loss, alongside them. Kept self-contained per this
project's convention rather than importing cross-phase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Small MLP projection head on top of frozen SigLIP embeddings.
    SigLIP itself is never fine-tuned -- this only maps its frozen 768-d
    output into a 128-d learned compatibility space."""

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Multiple-negatives-ranking (InfoNCE) loss. Identical to phase 9's --
    see that phase's model.py docstring for the false-negative-masking
    rationale."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def invariance_loss(z_orig, z_perturbed):
    """This phase's new loss term: 1 - mean cosine similarity between an
    item's projection from its original embedding and from its color-
    perturbed twin's embedding. Both inputs are already L2-normalized
    (ProjectionHead's forward() normalizes), so cosine similarity is just
    the dot product. Directly penalizes the model for encoding color as
    part of what makes two views "the same item.\""""
    cos_sim = (z_orig * z_perturbed).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, identical to phase 9's."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()

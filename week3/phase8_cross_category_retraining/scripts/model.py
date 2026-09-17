"""
Phase 8 shared model + loss code (identical to phase 7's -- same architecture,
same loss, only the training data differs: phase 8 trains on heterogeneous-
dyad-only positives instead of all also_buy edges). Copied rather than
imported cross-phase, consistent with this project's convention of keeping
each phase's folder self-contained so nothing has to be re-derived later.
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
    """Multiple-negatives-ranking (InfoNCE) loss.

    z_a: (B, D) anchor projections
    z_p: (B, D) positive projections
    z_extra: (B, K, D) explicit per-anchor negative projections (K = R random
        + H hard, already gathered per-row before calling this)
    inbatch_false_neg_mask: (B, B) bool, True at (i, j) i != j whenever
        batch-positive j is ALSO a true also_buy partner of anchor i --
        these must be masked out of the in-batch negative block, since at
        this pool's also_buy edge density a random batch member's positive
        can genuinely be another anchor's true partner too (a real risk, not
        a theoretical one, given the edge density measured in this project).
        Diagonal must be False (the true label for row i is column i).
    """
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau  # (B, B), column i = anchor i's own positive
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau  # (B, K)

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)  # (B, B+K)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic: mean pairwise cosine similarity across a
    batch of L2-normalized projections. A value drifting toward 1.0 over
    training indicates the projection head is collapsing toward a near-
    constant output rather than learning a useful compatibility space."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()

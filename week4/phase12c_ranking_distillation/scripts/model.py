"""
Phase 12 model code. `ProjectionHead` and `mnrl_loss` are copied unchanged
from phase 9's `model.py` (needed here only to load phase 9's existing
checkpoint for the baseline comparison -- no retraining of phase 9's model
happens in this phase). `ControllableProjectionHead` is new: this phase's own
controllable mode-embedding mechanism (see phase12_notes.md step 3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Unchanged from phase 8/9: frozen SigLIP 768-d -> 256 -> 128, L2-normalized."""

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


class ControllableProjectionHead(nn.Module):
    """One shared base projection (identical shape to phase 9's ProjectionHead:
    768 -> 256 -> 128) plus two learnable 128-d "mode" vectors. A forward pass
    takes a scalar alpha in [0, 1] (1.0 = pure substitute mode, 0.0 = pure
    complement mode, anything in between = an interpolated blend of the two
    mode vectors added to the same shared base projection) and L2-normalizes
    only at the end, after the mode vector is added -- the base projection
    itself is intentionally NOT normalized before that addition, so the mode
    vector has a consistent, comparable effect on the raw projection
    regardless of which mode is active.
    """

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.mode_substitute = nn.Parameter(torch.zeros(out_dim))
        self.mode_complement = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, alpha):
        """alpha: python float, or a (B,) tensor for per-example mode mixing."""
        base = self.net(x)
        if isinstance(alpha, torch.Tensor):
            mode_vec = alpha.unsqueeze(-1) * self.mode_substitute + (1 - alpha).unsqueeze(-1) * self.mode_complement
        else:
            mode_vec = alpha * self.mode_substitute + (1 - alpha) * self.mode_complement
        return F.normalize(base + mode_vec, p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Identical to phase 9's -- multiple-negatives-ranking (InfoNCE) loss,
    used here for the complement mode's training objective."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, unchanged from phase 9."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()

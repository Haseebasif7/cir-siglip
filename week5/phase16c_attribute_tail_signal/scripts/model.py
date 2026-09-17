"""Phase 16c model code. `ControllableProjectionHead` and `mnrl_loss` are
copied unchanged from phase 16's (which copied them from phase 12c) --
same architecture, same relevance-mode loss. Unlike phase 16, tail-exposure
mode also uses plain, UNWEIGHTED `mnrl_loss` this phase (no IPS weighting)
-- the new signal is which EDGES it trains on (attribute-based pairs
instead of also_buy edges), not how they're weighted. `uniformity_loss` is
ported in again as an optional, off-by-default safeguard, same as phase 16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControllableProjectionHead(nn.Module):
    """Unchanged from phase 16/12c: shared 768->256->128 projection plus two
    learnable 128-d mode vectors, blended by alpha and L2-normalized at the
    end. alpha=1.0 = pure relevance mode, alpha=0.0 = pure tail-exposure mode.
    """

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.mode_relevance = nn.Parameter(torch.zeros(out_dim))
        self.mode_tail = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, alpha):
        base = self.net(x)
        if isinstance(alpha, torch.Tensor):
            mode_vec = alpha.unsqueeze(-1) * self.mode_relevance + (1 - alpha).unsqueeze(-1) * self.mode_tail
        else:
            mode_vec = alpha * self.mode_relevance + (1 - alpha) * self.mode_tail
        return F.normalize(base + mode_vec, p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Identical to phase 9/12c/16's -- multiple-negatives-ranking (InfoNCE)
    loss. Used, unweighted, for BOTH modes this phase -- the difference
    between modes is entirely which edge population each is trained on."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, unchanged from phase 9/12c/16."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()


def uniformity_loss(f, t=2.0):
    """Ported from phase 13/16, OFF by default -- MNRL/softmax losses don't
    have the margin-loss collapse failure mode this was built for."""
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

"""
Phase 16 model code. `ControllableProjectionHead` and `mnrl_loss` are copied
unchanged (only the mode-vector names change) from phase 12c's
`ControllableProjectionHead` and `mnrl_loss` -- that architecture was already
proven, via real diagnostics, to produce a genuine smooth controllable dial
(phase 12c/12d), so it's reused as-is rather than redesigned. `alpha=1.0` is
"relevance" mode (mirrors phase 12c's substitute slot), `alpha=0.0` is
"tail-exposure" mode (mirrors phase 12c's complement slot).

`mnrl_loss_weighted` is new: identical to `mnrl_loss` except it applies a
per-example weight (this phase's inverse-propensity-scoring weight, see
`ips_weighting_check.md`) before averaging, instead of a plain mean over the
batch. This is the tail-exposure mode's training objective.

`uniformity_loss` is ported in from phase 13 as an optional, off-by-default
safeguard -- NOT part of phase 12c's own recipe (its MNRL/softmax loss
doesn't have the collapse failure mode a margin/hinge loss has), included
here only in case the smoke test (02_smoke_test.py) shows a collapse sign.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControllableProjectionHead(nn.Module):
    """One shared base projection (768 -> 256 -> 128) plus two learnable
    128-d mode vectors. A forward pass takes a scalar alpha in [0, 1]
    (1.0 = pure relevance mode, 0.0 = pure tail-exposure mode, anything in
    between = an interpolated blend of the two mode vectors added to the
    same shared base projection) and L2-normalizes only at the end, after
    the mode vector is added -- unchanged from phase 12c's
    ControllableProjectionHead, only the mode-vector names differ.
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
        """alpha: python float, or a (B,) tensor for per-example mode mixing."""
        base = self.net(x)
        if isinstance(alpha, torch.Tensor):
            mode_vec = alpha.unsqueeze(-1) * self.mode_relevance + (1 - alpha).unsqueeze(-1) * self.mode_tail
        else:
            mode_vec = alpha * self.mode_relevance + (1 - alpha) * self.mode_tail
        return F.normalize(base + mode_vec, p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Identical to phase 9/12c's -- multiple-negatives-ranking (InfoNCE)
    loss, used here for the relevance mode's training objective, unweighted."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mnrl_loss_weighted(z_a, z_p, z_extra, inbatch_false_neg_mask, weights, tau=0.07):
    """Same construction as mnrl_loss, but each anchor's cross-entropy term
    is scaled by its IPS weight before averaging, instead of a plain batch
    mean. Used for the tail-exposure mode's training objective. `weights`:
    (B,) tensor, already capped and mean-1-rescaled (see
    01_prepare_training_data.py)."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    per_example_loss = F.cross_entropy(logits, labels, reduction="none")
    return (weights * per_example_loss).mean()


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, unchanged from phase 9/12c."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()


def uniformity_loss(f, t=2.0):
    """Ported from phase 13 (Wang & Isola 2020 alignment/uniformity
    regularizer), OFF by default here -- ControllableProjectionHead's
    MNRL/softmax losses don't have the margin-loss collapse failure mode
    this was built for, so it's included only as a safeguard the smoke test
    can enable if it turns out to be needed, not part of phase 12c's own
    recipe. f: (N, D) L2-normalized embeddings from one batch."""
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

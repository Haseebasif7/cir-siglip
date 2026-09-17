"""
Phase 19: the simplest possible learned mechanism on top of frozen SigLIP --
a single element-wise weight vector, same dimensionality as the embedding
(768), applied identically regardless of whether an item is used as a
query-context item or a candidate. No bias term, no cross-dimension mixing
(a Linear layer would mix dimensions; this deliberately doesn't), no
category or role conditioning. Per the brief: this needs to stay
deliberately simple, since the whole point of this step is testing this
mechanism on its own before phase 19's later projection/attention steps add
real capacity.

`mnrl_loss` is copied unchanged from phase 9 / phase 17's own model.py
(`week3/phase9_polyvore_compatibility/scripts/model.py`,
`week5/phase17_dedicated_capacity_generalization/scripts/model.py`) -- same
loss, same in-batch false-negative masking convention, reused verbatim
since this phase trains on the identical compatibility signal those phases
did (real Polyvore outfit co-occurrence, complement-mode style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_DIM = 768


class FeatureWeighting(nn.Module):
    def __init__(self, dim=SIGLIP_DIM):
        super().__init__()
        # ones-init: at init this is an exact identity map on the L2-normalized
        # input (x * 1 renormalized == x), the multiplicative equivalent of
        # this project's standard zero-init-additive convention (e.g. phase
        # 12c/16d's mode vectors) -- training starts from a true no-op, not
        # an arbitrary random rescaling.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x * self.weight, p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Multiple-negatives-ranking (InfoNCE) loss, identical to phase 9/17's.

    z_a: (B, D) anchor projections
    z_p: (B, D) positive projections
    z_extra: (B, K, D) explicit per-anchor RANDOM negative projections (no
        hard-negative mining -- see phase 19's brief: this project has now
        found mined hard negatives underperform random ones in three
        separate contexts, phases 7-9, 13c, and 14b's first run).
    inbatch_false_neg_mask: (B, B) bool, True at (i, j) i != j whenever
        batch-positive j is ALSO a true positive partner of anchor i.
    """
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, identical to phase 9/17's."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()

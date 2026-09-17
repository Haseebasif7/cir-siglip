"""
Phase 21: `ProjectionHead` copied unchanged from phase 9's model.py. Used
TWICE in this phase, as two fully separate, independently-parameterized
instances (one for the complement head, one for the substitute head) --
there is no shared module, no shared layer, no shared parameters of any
kind between the two. Blending happens purely at inference time (see
`04_alpha_sweep.py`), never inside this class -- unlike every prior
controllable-dial phase in this project (12c/16d/17/18/18b), which all had
some notion of a shared trunk or shared base layer that alpha conditioned.
`mnrl_loss` (complement head's loss) and `mean_pairwise_cosine` (collapse
diagnostic) are also copied unchanged from phase 9/17's model.py.
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


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Multiple-negatives-ranking (InfoNCE) loss, identical to phase 9/17's."""
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

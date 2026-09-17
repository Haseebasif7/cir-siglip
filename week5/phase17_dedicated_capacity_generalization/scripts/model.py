"""
Phase 17 model. `DedicatedCapacityHead` is phase 16d's dedicated-capacity
architecture (week5/phase16d_dedicated_capacity/scripts/model.py), retrofit
onto the substitute/complement axis: a minimal shared 768->256 dimensionality
reduction feeding two fully independent 256->128 heads, one per mode, each
independently L2-normalized before blending by alpha and renormalizing.
Renamed `mode_relevance`/`mode_tail` -> `substitute_head`/`complement_head`
to match this mechanism's own naming (alpha=1.0 = pure substitute, alpha=0.0
= pure complement, matching phase 12/12b/12c's convention exactly).

`mnrl_loss` and `mean_pairwise_cosine` are unchanged copies (phase 12c's
`model.py`), needed for the complement-mode loss and the collapse-guard
diagnostic respectively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DedicatedCapacityHead(nn.Module):
    """Minimal shared trunk (768->256, dimensionality reduction only) feeding
    two fully independent 256->128 heads. No small additive correction vector
    (phase 12/12b/12c's design) -- each mode gets its own real transformation.
    """

    def __init__(self, in_dim=768, shared_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.substitute_head = nn.Linear(shared_dim, out_dim)
        self.complement_head = nn.Linear(shared_dim, out_dim)

    def forward(self, x, alpha):
        """alpha: python float, or a (B,) tensor for per-example mode mixing.
        alpha=1.0 -> pure substitute, alpha=0.0 -> pure complement."""
        h = self.shared(x)
        z_sub = F.normalize(self.substitute_head(h), p=2, dim=-1)
        z_comp = F.normalize(self.complement_head(h), p=2, dim=-1)
        if isinstance(alpha, torch.Tensor):
            blend = alpha.unsqueeze(-1) * z_sub + (1 - alpha).unsqueeze(-1) * z_comp
        else:
            blend = alpha * z_sub + (1 - alpha) * z_comp
        return F.normalize(blend, p=2, dim=-1)

    def head_outputs(self, x):
        h = self.shared(x)
        z_sub = F.normalize(self.substitute_head(h), p=2, dim=-1)
        z_comp = F.normalize(self.complement_head(h), p=2, dim=-1)
        return z_sub, z_comp


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Identical to phase 9/12c's -- multiple-negatives-ranking (InfoNCE)
    loss, used here for the complement mode's training objective."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))

    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau

    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, unchanged from phase 9/12c."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()

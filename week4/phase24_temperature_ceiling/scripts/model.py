"""Phase 23: ProjectionHead/mnrl_loss/mean_pairwise_cosine, copied unchanged
from phase 9/12's model.py -- no architecture changes in this phase (explicitly
out of scope per the brief's "do not touch model architecture" instruction)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
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
    """Unchanged from phase 7/8/9: in-batch negatives (masked for false
    negatives) + explicit extra negatives, softmax cross-entropy."""
    B = z_a.shape[0]
    inbatch_logits = (z_a @ z_p.T) / tau
    inbatch_logits = inbatch_logits.masked_fill(inbatch_false_neg_mask, float("-inf"))
    extra_logits = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
    logits = torch.cat([inbatch_logits, extra_logits], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def mean_pairwise_cosine(z, n_sample=256):
    n = min(n_sample, z.shape[0])
    idx = torch.randperm(z.shape[0], device=z.device)[:n]
    zs = z[idx]
    sims = zs @ zs.T
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return sims[mask].mean().item()

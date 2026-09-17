"""
Phase 20: `ProjectionHead` copied unchanged from phase 9/12's model.py --
needed only to load phase 9's existing checkpoint (`model_a_random_negs.pt`)
for direct re-verification. No retraining happens in this phase.
"""
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

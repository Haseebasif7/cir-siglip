"""
Phase 27 shared model code. ProjectionHeadGeneral is copied unchanged from
phase 25/26 (in_dim already a constructor parameter there, so no new class
is needed to widen it to 1536 or 2304 -- consistent with this project's
convention of keeping each phase's folder self-contained).

CATEGORY_LIST fixes the index order for both category-conditioning variants
(the learned embedding table, and the SigLIP-phrase lookup) -- must match
00_extract_text_embeddings.py's CATEGORY_LIST exactly, and matches the CIR
benchmark's own 11 pool categories.
"""
import torch.nn as nn
import torch.nn.functional as F

CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}


class ProjectionHeadGeneral(nn.Module):
    def __init__(self, in_dim=768, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, mask, tau):
    """Identical to phase 9/23/25/26's loss -- copied, not imported, per
    this project's cross-phase convention."""
    import torch
    B = z_a.shape[0]
    inbatch_logits = (z_a @ z_p.T) / tau
    inbatch_logits = inbatch_logits.masked_fill(mask, float("-inf"))
    extra_logits = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
    logits = torch.cat([inbatch_logits, extra_logits], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)

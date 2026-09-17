"""
Phase 33: CSA-Net's category-pair-conditioned subspace attention mechanism,
copied UNCHANGED from
week4/phase13b_csa_net_siglip_backbone/scripts/model.py -- `siglip_dim` was
already a constructor parameter, so text integration (step 2) needs zero
architecture code changes here, exactly the same situation phase 31 found
for OutfitTransformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CATEGORIES = 11
NUM_SUBSPACES = 5
EMBED_DIM = 64
SIGLIP_DIM = 768  # 1536 with text input (step 2)
MARGIN = 0.3
ATTN_HIDDEN = 64


class CSANetSigLIP(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, num_subspaces=NUM_SUBSPACES,
                 embed_dim=EMBED_DIM, attn_hidden=ATTN_HIDDEN, siglip_dim=SIGLIP_DIM):
        super().__init__()
        self.proj = nn.Linear(siglip_dim, embed_dim)

        self.attn_net = nn.Sequential(
            nn.Linear(2 * num_categories, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, num_subspaces),
        )

        self.masks = nn.Parameter(torch.ones(num_subspaces, embed_dim) + torch.randn(num_subspaces, embed_dim) * 0.01)

        self.num_categories = num_categories
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim

    def encode_feature(self, siglip_vecs):
        """siglip_vecs: (..., siglip_dim) -> (..., embed_dim). nn.Linear and
        F.normalize(dim=-1) both operate on the last dim only, so this
        supports arbitrary leading batch dims natively (used directly on
        (B, Lmax, D) and (B, R, D) shaped inputs in the vectorized training
        loop, not just flat (N, D))."""
        return self.proj(F.normalize(siglip_vecs, p=2, dim=-1))

    def attention_weights(self, cat_s_onehot, cat_t_onehot):
        logits = self.attn_net(torch.cat([cat_s_onehot, cat_t_onehot], dim=-1))
        return F.softmax(logits, dim=-1)

    def embed_from_feature(self, x, cat_s_onehot, cat_t_onehot):
        """x: (N, embed_dim) FLAT batch (the training loop flattens any
        extra (B, Lmax, ...) dims to N before calling this, and reshapes
        the result back -- kept 2D-only here, matching phase 13/13b/31's own
        convention of not modifying architecture code for a training-loop
        vectorization)."""
        w = self.attention_weights(cat_s_onehot, cat_t_onehot)  # (N, k)
        masked = x.unsqueeze(1) * self.masks.unsqueeze(0)  # (N, k, embed_dim)
        f = (masked * w.unsqueeze(-1)).sum(dim=1)  # (N, embed_dim)
        return F.normalize(f, p=2, dim=-1)

    def forward(self, siglip_vecs, cat_s_onehot, cat_t_onehot):
        x = self.encode_feature(siglip_vecs)
        return self.embed_from_feature(x, cat_s_onehot, cat_t_onehot)

    def all_category_embeddings_from_feature(self, x, cat_s_onehot):
        return self._all_category_embeddings(x, cat_s_onehot, enumerate_slot="t")

    def all_as_candidate_embeddings_from_feature(self, x, cat_t_onehot):
        return self._all_category_embeddings(x, cat_t_onehot, enumerate_slot="s")

    def _all_category_embeddings(self, x, fixed_onehot, enumerate_slot):
        B, C = fixed_onehot.shape
        k = self.num_subspaces
        other_all = torch.eye(C, device=x.device, dtype=fixed_onehot.dtype)
        fixed_rep = fixed_onehot.unsqueeze(1).expand(B, C, C).reshape(B * C, C)
        other_rep = other_all.unsqueeze(0).expand(B, C, C).reshape(B * C, C)
        if enumerate_slot == "t":
            w = self.attention_weights(fixed_rep, other_rep)
        else:
            w = self.attention_weights(other_rep, fixed_rep)
        w = w.view(B, C, k)
        masked = x.unsqueeze(1).unsqueeze(1) * self.masks.view(1, 1, k, -1)
        f = (masked * w.unsqueeze(-1)).sum(dim=2)
        return F.normalize(f, p=2, dim=-1)


def pairwise_distance(a, b):
    return ((a - b) ** 2).sum(dim=-1)


def outfit_ranking_loss(d_pos, d_negs, margin=MARGIN, aggregation="min"):
    if aggregation == "min":
        d_neg_agg = d_negs.min(dim=-1).values
    elif aggregation == "mean":
        d_neg_agg = d_negs.mean(dim=-1)
    else:
        raise ValueError(aggregation)
    loss = F.relu(d_pos - d_neg_agg + margin)
    return loss.mean()


def uniformity_loss(f, t=2.0):
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

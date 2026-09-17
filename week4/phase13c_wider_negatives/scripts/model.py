"""
Phase 13c: byte-for-byte identical architecture to phase 13b's `model.py`
(CSA-Net's category-pair-conditioned subspace attention on a FROZEN SigLIP
backbone). Copied unchanged -- this phase's one change (NUM_NEGATIVES 10->20)
lives entirely in train_core.py, not here. See phase 13b's
architecture_notes.md for the full architecture trace and phase 13c's own
../phase13c_notes.md for what this phase changed and why.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CATEGORIES = 11
NUM_SUBSPACES = 5  # k=5, paper section 4.2, unchanged from phase 13
EMBED_DIM = 64  # unchanged from phase 13
SIGLIP_DIM = 768  # phase 9's embeddings/siglip_base.npz
MARGIN = 0.3  # unchanged from phase 13
ATTN_HIDDEN = 64  # unchanged from phase 13


class CSANetSigLIP(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, num_subspaces=NUM_SUBSPACES,
                 embed_dim=EMBED_DIM, attn_hidden=ATTN_HIDDEN, siglip_dim=SIGLIP_DIM):
        super().__init__()
        self.proj = nn.Linear(siglip_dim, embed_dim)  # replaces phase 13's Linear(512, 64)

        self.attn_net = nn.Sequential(
            nn.Linear(2 * num_categories, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, num_subspaces),
        )

        # Same near-identity init as phase 13, for the same reason (dead-gradient
        # fix) -- verified again in this phase rather than assumed, see
        # ../training_log.md step 2.
        self.masks = nn.Parameter(torch.ones(num_subspaces, embed_dim) + torch.randn(num_subspaces, embed_dim) * 0.01)

        self.num_categories = num_categories
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim

    def encode_feature(self, siglip_vecs):
        """siglip_vecs: (B, 768) frozen SigLIP embeddings (already L2-normalized
        at extraction time in phase 9, but re-normalizing here is harmless and
        makes this function's contract self-contained) -> x: (B, embed_dim)."""
        return self.proj(F.normalize(siglip_vecs, p=2, dim=-1))

    def attention_weights(self, cat_s_onehot, cat_t_onehot):
        logits = self.attn_net(torch.cat([cat_s_onehot, cat_t_onehot], dim=-1))
        return F.softmax(logits, dim=-1)

    def embed_from_feature(self, x, cat_s_onehot, cat_t_onehot):
        w = self.attention_weights(cat_s_onehot, cat_t_onehot)  # (B, k)
        masked = x.unsqueeze(1) * self.masks.unsqueeze(0)  # (B, k, embed_dim)
        f = (masked * w.unsqueeze(-1)).sum(dim=1)  # (B, embed_dim)
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
    """Unchanged from phase 13 -- see that file's docstring for the full
    derivation and the collapse evidence that motivated it. Re-verified (not
    assumed) to still be needed in this phase -- see training_log.md step 2."""
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

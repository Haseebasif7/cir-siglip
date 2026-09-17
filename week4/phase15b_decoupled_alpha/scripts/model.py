"""
Phase 15b: architecture UNCHANGED from phase 15's `CSANetSigLIPControllable`
(copied verbatim -- see `week4/phase15_controllable_subspace_attention/scripts/model.py`
for the original design-decision log). Only the TRAINING PROCEDURE changes
this phase (see train_core.py): alpha is removed from the loss-weighting
envelope entirely, replaced by a fixed dual forward-pass (complement always
at alpha=0, substitute always at alpha=1, every step, both trained
simultaneously with fixed weights) -- this directly targets the "alpha
double-duty confound" phase 15 diagnosed (alpha serving as both the
attn_net conditioning input and the outer loss-term weight).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CATEGORIES = 11
NUM_SUBSPACES = 5  # k=5, paper section 4.2, unchanged from phase 13/13b/15
EMBED_DIM = 64  # unchanged from phase 13/13b/15
SIGLIP_DIM = 768  # phase 9's embeddings/siglip_base.npz
MARGIN = 0.3  # unchanged from phase 13/13b/15
ATTN_HIDDEN = 64  # unchanged from phase 13/13b/15


class CSANetSigLIPControllable(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, num_subspaces=NUM_SUBSPACES,
                 embed_dim=EMBED_DIM, attn_hidden=ATTN_HIDDEN, siglip_dim=SIGLIP_DIM):
        super().__init__()
        self.proj = nn.Linear(siglip_dim, embed_dim)

        self.attn_net = nn.Sequential(
            nn.Linear(2 * num_categories + 1, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, num_subspaces),
        )

        self.masks = nn.Parameter(torch.ones(num_subspaces, embed_dim) + torch.randn(num_subspaces, embed_dim) * 0.01)

        self.num_categories = num_categories
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim

    def encode_feature(self, siglip_vecs):
        """siglip_vecs: (B, 768) frozen SigLIP embeddings -> x: (B, embed_dim)."""
        return self.proj(F.normalize(siglip_vecs, p=2, dim=-1))

    @staticmethod
    def _alpha_column(alpha, batch, device, dtype):
        """Broadcast alpha (python float, or a (B,)/(1,) tensor) to a (batch, 1) column."""
        if isinstance(alpha, torch.Tensor):
            a = alpha.to(device=device, dtype=dtype).reshape(-1, 1)
            if a.shape[0] == 1 and batch > 1:
                a = a.expand(batch, 1)
            assert a.shape[0] == batch, f"alpha batch {a.shape[0]} != {batch}"
            return a
        return torch.full((batch, 1), float(alpha), device=device, dtype=dtype)

    def attention_weights(self, cat_s_onehot, cat_t_onehot, alpha):
        batch = cat_s_onehot.shape[0]
        a = self._alpha_column(alpha, batch, cat_s_onehot.device, cat_s_onehot.dtype)
        logits = self.attn_net(torch.cat([cat_s_onehot, cat_t_onehot, a], dim=-1))
        return F.softmax(logits, dim=-1)

    def embed_from_feature(self, x, cat_s_onehot, cat_t_onehot, alpha):
        w = self.attention_weights(cat_s_onehot, cat_t_onehot, alpha)  # (B, k)
        masked = x.unsqueeze(1) * self.masks.unsqueeze(0)  # (B, k, embed_dim)
        f = (masked * w.unsqueeze(-1)).sum(dim=1)  # (B, embed_dim)
        return F.normalize(f, p=2, dim=-1)

    def forward(self, siglip_vecs, cat_s_onehot, cat_t_onehot, alpha):
        x = self.encode_feature(siglip_vecs)
        return self.embed_from_feature(x, cat_s_onehot, cat_t_onehot, alpha)

    def all_category_embeddings_from_feature(self, x, cat_s_onehot, alpha):
        return self._all_category_embeddings(x, cat_s_onehot, enumerate_slot="t", alpha=alpha)

    def all_as_candidate_embeddings_from_feature(self, x, cat_t_onehot, alpha):
        return self._all_category_embeddings(x, cat_t_onehot, enumerate_slot="s", alpha=alpha)

    def _all_category_embeddings(self, x, fixed_onehot, enumerate_slot, alpha):
        B, C = fixed_onehot.shape
        k = self.num_subspaces
        other_all = torch.eye(C, device=x.device, dtype=fixed_onehot.dtype)
        fixed_rep = fixed_onehot.unsqueeze(1).expand(B, C, C).reshape(B * C, C)
        other_rep = other_all.unsqueeze(0).expand(B, C, C).reshape(B * C, C)
        if enumerate_slot == "t":
            w = self.attention_weights(fixed_rep, other_rep, alpha)
        else:
            w = self.attention_weights(other_rep, fixed_rep, alpha)
        w = w.view(B, C, k)
        masked = x.unsqueeze(1).unsqueeze(1) * self.masks.view(1, 1, k, -1)
        f = (masked * w.unsqueeze(-1)).sum(dim=2)
        return F.normalize(f, p=2, dim=-1)

    def attention_weight_shift(self, cat_s_onehot, cat_t_onehot):
        """Same probe as phase 15: fixed sample of (cat_s, cat_t) pairs, sweep
        the forward-pass alpha alone from 0 to 1 (no loss), mean L1 distance
        between the alpha=0 and alpha=1 attention-weight vectors."""
        with torch.no_grad():
            w0 = self.attention_weights(cat_s_onehot, cat_t_onehot, 0.0)
            w1 = self.attention_weights(cat_s_onehot, cat_t_onehot, 1.0)
            return (w0 - w1).abs().sum(dim=-1).mean().item()


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
    """Unchanged from phase 13/13b/15 -- Wang & Isola (2020) alignment/
    uniformity regularizer."""
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())


def ranking_distillation_loss(z_anchor, z_neighbors, teacher_sims, tau=0.07):
    """Phase 12c's substitute-mode KL(teacher || student) ranking-distillation
    loss, adapted to take already-computed CSA-Net embeddings.
    z_anchor: (B, embed_dim) L2-normalized anchor embeddings.
    z_neighbors: (B, K, embed_dim) L2-normalized same-category neighbor embeddings.
    teacher_sims: (B, K) raw-SigLIP cosine similarities (from nn_lookup.npz).
    """
    teacher_dist = F.softmax(teacher_sims / tau, dim=-1)
    student_sims = torch.einsum("bd,bkd->bk", z_anchor, z_neighbors)
    student_log_probs = F.log_softmax(student_sims / tau, dim=-1)
    return F.kl_div(student_log_probs, teacher_dist, reduction="batchmean")

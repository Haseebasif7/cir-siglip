"""
Phase 13: CSA-Net architecture, reproduced from Lin, Tran & Davis, "Fashion
Outfit Complementary Item Retrieval" (CVPR 2020), sections 3.1-3.2. See
../implementation_notes.md for the full paper-vs-implementation trace;
comments here only cover choices not obvious from the paper text itself.

Architecture (paper Figure 2):
  image -> ResNet18 (ImageNet-pretrained) -> 512-d pooled feature
         -> Linear(512, 64) -> x  (the "image feature vector", embedding size
            64, matched to the paper's fair-comparison setting)
  (source category one-hot, target category one-hot) -> concat -> FC -> ReLU
         -> FC -> softmax -> attention weights w_1..w_k  (k=5 subspaces)
  k learnable masks m_1..m_k, same dim as x (64)
  f = sum_i (x ⊙ m_i) * w_i        <- final embedding, eq. (1)
  f <- f / ||f||_2                 <- L2-normalize -- NOT in the paper's own
       equation 1, added after observing empirically (real training run)
       that the un-normalized hinge/ranking loss has a trivial degenerate
       solution: shrink every embedding toward the origin so D_pos and
       D_neg both -> 0 together, which drives the loss to exactly the
       margin value without learning anything about relative compatibility.
       Both D_pos and D_neg were observed shrinking in lockstep across two
       independent training runs (real Modal GPU run and a local smoke
       test) -- confirming the collapse, not just a hypothesis. Bounding
       ||f||=1 makes that trivial solution unavailable, forcing the network
       to actually separate positive from negative distances to reduce the
       loss. This is also standard practice in the closely related papers
       CSA-Net itself builds on (Type-aware, SCE-Net) and matches every
       other phase in this project (7-12), which all L2-normalize their
       final embeddings before computing similarity/distance.

Loss (paper section 3.2, eq. 5-7): outfit ranking loss, aggregation="min"
(paper's own best setting, Table 3), margin 0.3 (paper's stated value).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

NUM_CATEGORIES = 11
NUM_SUBSPACES = 5  # k=5, paper section 4.2 ("we set the number of subspaces to 5 as in [15]")
EMBED_DIM = 64  # paper: "embedding size 64 ... for fair comparison" with Type-aware/SCE-Net
MARGIN = 0.3  # paper section 4.2
ATTN_HIDDEN = 64  # paper only specifies "two fully connected layers + softmax", not the
                   # hidden width -- 64 (= EMBED_DIM) chosen as a reasonable unstated default,
                   # documented in implementation_notes.md as an assumption.


class CSANet(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, num_subspaces=NUM_SUBSPACES,
                 embed_dim=EMBED_DIM, attn_hidden=ATTN_HIDDEN, pretrained=True):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()  # keep the 512-d pooled feature
        self.backbone = backbone
        self.proj = nn.Linear(512, embed_dim)

        self.attn_net = nn.Sequential(
            nn.Linear(2 * num_categories, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, num_subspaces),
        )  # softmax applied in forward, not here, so raw logits are available if ever needed

        # Paper doesn't specify mask initialization. randn*0.01 (tried first)
        # makes f = sum_i (x*m_i)*w_i near-ZERO at init regardless of x --
        # verified empirically: with softmax attention weights summing to 1,
        # near-identity masks (~1 +- small noise) instead make f ~= x at
        # init, so the ranking loss has real gradient signal from step 1
        # (built on the pretrained backbone's own already-somewhat-
        # discriminative features) instead of starting from a flat, near-
        # zero embedding for every item -- this only changes the starting
        # point, not what the paper's own loss/architecture can converge to.
        # Also now numerically necessary given L2-normalization below (see
        # module docstring): normalizing a near-zero-norm vector amplifies
        # whatever tiny numerical noise it has into an essentially random
        # unit direction, not a meaningful one.
        self.masks = nn.Parameter(torch.ones(num_subspaces, embed_dim) + torch.randn(num_subspaces, embed_dim) * 0.01)

        self.num_categories = num_categories
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim

    def encode_image(self, images):
        """images: (B, 3, 224, 224) -> x: (B, embed_dim), the base CNN feature
        BEFORE any subspace masking. Cached/reused across category-pairs since
        it doesn't depend on category vectors at all (paper Figure 2)."""
        feat = self.backbone(images)
        return self.proj(feat)

    def attention_weights(self, cat_s_onehot, cat_t_onehot):
        """(B, C) x2 -> (B, k) softmax attention weights over subspaces."""
        logits = self.attn_net(torch.cat([cat_s_onehot, cat_t_onehot], dim=-1))
        return F.softmax(logits, dim=-1)

    def embed_from_feature(self, x, cat_s_onehot, cat_t_onehot):
        """x: (B, embed_dim) precomputed base feature. Applies subspace
        masking + attention weighting -- eq. (1). Split out from forward()
        so the (expensive) CNN pass and (cheap) category-conditioning can be
        reused/recombined independently, matching the paper's own indexing
        scheme (Section 3.3): a single CNN pass per image, many
        category-pair-conditioned embeddings derived cheaply from it."""
        w = self.attention_weights(cat_s_onehot, cat_t_onehot)  # (B, k)
        masked = x.unsqueeze(1) * self.masks.unsqueeze(0)  # (B, k, embed_dim)
        f = (masked * w.unsqueeze(-1)).sum(dim=1)  # (B, embed_dim)
        return F.normalize(f, p=2, dim=-1)

    def forward(self, images, cat_s_onehot, cat_t_onehot):
        x = self.encode_image(images)
        return self.embed_from_feature(x, cat_s_onehot, cat_t_onehot)

    def all_category_embeddings_from_feature(self, x, cat_s_onehot):
        """x: (B, embed_dim), cat_s_onehot: (B, C) -- the item's OWN category,
        fixed IN THE FIRST (cat_s) ARGUMENT SLOT. Returns (B, C, embed_dim):
        the item's embedding for every possible value of the SECOND (cat_t)
        argument slot, computed in one batched call. Used for CONTEXT/QUERY
        items at eval time -- eq. 2 always puts the query item's own category
        in the first slot and the (fixed-per-query) target category in the
        second, so caching over the second slot covers every query this item
        could appear in."""
        return self._all_category_embeddings(x, cat_s_onehot, enumerate_slot="t")

    def all_as_candidate_embeddings_from_feature(self, x, cat_t_onehot):
        """x: (B, embed_dim), cat_t_onehot: (B, C) -- the item's OWN category,
        fixed IN THE SECOND (cat_t) ARGUMENT SLOT. Returns (B, C, embed_dim):
        the item's embedding for every possible value of the FIRST (cat_s)
        argument slot. Used for CANDIDATE/POOL items at eval time -- eq. 3/4
        always put the query context item's category in the first slot and
        the candidate's own (pool) category in the second, so a candidate
        needs one embedding per possible context-item category it might be
        compared against, not per target category (its target category is
        always its own, fixed by which pool it's in)."""
        return self._all_category_embeddings(x, cat_t_onehot, enumerate_slot="s")

    def _all_category_embeddings(self, x, fixed_onehot, enumerate_slot):
        B, C = fixed_onehot.shape
        k = self.num_subspaces
        other_all = torch.eye(C, device=x.device, dtype=fixed_onehot.dtype)  # (C, C)
        fixed_rep = fixed_onehot.unsqueeze(1).expand(B, C, C).reshape(B * C, C)
        other_rep = other_all.unsqueeze(0).expand(B, C, C).reshape(B * C, C)
        if enumerate_slot == "t":
            w = self.attention_weights(fixed_rep, other_rep)  # cat_s=fixed, cat_t=enumerated
        else:
            w = self.attention_weights(other_rep, fixed_rep)  # cat_s=enumerated, cat_t=fixed
        w = w.view(B, C, k)
        masked = x.unsqueeze(1).unsqueeze(1) * self.masks.view(1, 1, k, -1)  # (B,1,k,D)
        f = (masked * w.unsqueeze(-1)).sum(dim=2)  # (B, C, D)
        return F.normalize(f, p=2, dim=-1)


def pairwise_distance(a, b):
    """Squared Euclidean distance, paper doesn't specify the exact metric
    beyond "pairwise distance" (d(.,.) in eq. 5) -- squared L2 is the
    standard choice for triplet/ranking losses in this line of work
    (Type-aware, SCE-Net both use it), used here as the documented
    assumption. a, b: (..., D) -> (...,)"""
    return ((a - b) ** 2).sum(dim=-1)


def outfit_ranking_loss(d_pos, d_negs, margin=MARGIN, aggregation="min"):
    """d_pos: (B,) distance of outfit to positive item.
    d_negs: (B, M) distances of outfit to M negative items.
    aggregation: "min" or "mean" over the M negatives -- paper Table 3 found
    "min" (i.e. the single hardest negative) works best; used as the default.
    Returns per-sample hinge loss (eq. 7), mean-reduced.
    """
    if aggregation == "min":
        d_neg_agg = d_negs.min(dim=-1).values
    elif aggregation == "mean":
        d_neg_agg = d_negs.mean(dim=-1)
    else:
        raise ValueError(aggregation)
    loss = F.relu(d_pos - d_neg_agg + margin)
    return loss.mean()


def uniformity_loss(f, t=2.0):
    """NOT part of the paper -- added after four independent real-data
    training runs (see training_log.md) all showed the SAME failure mode:
    D_pos and D_neg both shrinking together toward 0 (embeddings collapsing
    to nearly the same point on the unit hypersphere, confirmed directly --
    a squared-distance of ~0.04 between L2-normalized vectors implies cosine
    similarity ~0.98 between essentially unrelated items). This is a known,
    structural weakness of margin/hinge losses specifically (unlike a
    softmax/cross-entropy-based loss, a hinge loss has no term that
    penalizes ALL embeddings becoming similar -- collapsing every item
    (positive AND negative) toward the same point trivially drives both
    D_pos and D_neg_agg to 0 together, satisfying `D_pos - D_neg_agg +
    margin <= 0` for margin values below a small threshold... actually
    reaching exactly the margin's own value as the residual, matching what
    was observed in every real run). This adds Wang & Isola's (2020)
    alignment/uniformity regularizer -- log(mean(exp(-t*||f_i-f_j||^2))) for
    random pairs in the batch -- to the loss with a small weight. It is
    MINIMIZED when embeddings spread across the hypersphere and MAXIMIZED
    (worst) exactly when they collapse, giving the optimizer no free lunch
    for the collapse solution while not changing what a genuinely
    compatibility-discriminative embedding looks like (a real, well-trained
    embedding is naturally close to uniform on the sphere for unrelated
    items already). f: (N, D) L2-normalized embeddings from one batch."""
    n = f.shape[0]
    sq_dists = torch.cdist(f, f, p=2) ** 2  # (N, N)
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

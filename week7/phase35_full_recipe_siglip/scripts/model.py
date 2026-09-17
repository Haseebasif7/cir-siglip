"""
Phase 35: OutfitTransformerSigLIP architecture, extended from phase 31/32's
own copy (week7/phase32_partial_ensemble_outfittransformer/scripts/model.py,
itself copied unchanged from week4/phase14b's original). The base
transformer set-encoder, encode_item_tokens, and embed_item_alone are
BYTE-IDENTICAL to phase 32's -- this phase's whole point is to isolate what
the training RECIPE adds on top of an unchanged architecture+backbone, so
nothing about the shared machinery may change.

Two additions, both read straight from the paper (arXiv 2204.04812), not
from memory -- see ../implementation_notes.md for the exact quotes and the
one deliberate adaptation each required:

1. `cp_head`: a single Linear(d_model, 1) producing a raw CP (compatibility
   prediction) logit from the outfit token's output position -- stage 1
   only, unused after that (its weights are not warm-started into stage 2).

2. `embed_set`: a generalization of phase 14b/31/32's `embed_query` that
   accepts an arbitrary "lead token" to prepend instead of hardcoding
   `self.outfit_token`. Stage 1 (CP) passes `self.outfit_token` (a global
   "is this a real outfit" summary, unchanged from phase 14b/31/32's own
   role for that token). Stage 2 (targeted retrieval) passes a
   target-CATEGORY token instead -- this is piece 2 of the brief, and is
   exactly the role the paper's own "target item token s" plays: "The
   transformer encoder takes as input the set of feature vectors F of the
   partial outfit, and the target item specification s ... t =
   MLP(E_trans(s, F))" (paper section 3.2). `embed_query` (old name, still
   present) and `embed_query_targeted` (new) are both thin wrappers over
   `embed_set` for clarity at call sites.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_DIM = 1536        # image+text (phase 32's winning input_mode) -- see implementation_notes.md
D_MODEL = 128
D_EMBED = 64
N_HEADS = 8
N_LAYERS = 4
D_FFN = 512
DROPOUT = 0.1
MARGIN = 0.2              # phase 31/32's tuned winner, held constant here (not re-tuned, see brief)
UNIFORMITY_WEIGHT = 0.1    # phase 31/32's tuned winner, held constant here


class OutfitTransformerSigLIP(nn.Module):
    def __init__(self, siglip_dim=SIGLIP_DIM, d_model=D_MODEL, d_embed=D_EMBED,
                 n_heads=N_HEADS, n_layers=N_LAYERS, d_ffn=D_FFN, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(siglip_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True, activation=F.mish,
        )
        self.set_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.embed_ffn = nn.Linear(d_model, d_embed, bias=False)
        self.outfit_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.cp_head = nn.Linear(d_model, 1)  # stage 1 only (piece 1)

    def encode_item_tokens(self, siglip_vecs):
        """siglip_vecs: (N, siglip_dim) raw features (image+text concat, any
        norm -- including the synthetic "empty image || category text"
        vectors used to build target-category tokens, see
        build_category_tokens below). Returns (N, d_model) item tokens,
        L2-normalized. Unchanged from phase 32."""
        x = F.normalize(siglip_vecs, p=2, dim=-1)
        e = self.proj(x)
        return F.normalize(e, p=2, dim=-1)

    def embed_set(self, lead_token, ctx_tokens, pad_mask):
        """lead_token: (d_model,) [broadcast to the whole batch] or
        (B, d_model) [one per sample -- used for per-sample target-category
        tokens in stage 2]. ctx_tokens: (B, L, d_model) already-projected+
        normalized tokens. pad_mask: (B, L) bool, True = padded. Returns the
        RAW (B, d_model) transformer output at the lead token's position
        (position 0) -- callers apply their own head (embed_ffn for
        retrieval, cp_head for CP)."""
        B = ctx_tokens.shape[0]
        if lead_token.dim() == 1:
            lead = lead_token.view(1, 1, -1).expand(B, -1, -1)
        else:
            lead = lead_token.view(B, 1, -1)
        seq = torch.cat([lead, ctx_tokens], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device), pad_mask], dim=1)
        h = self.set_enc(seq, src_key_padding_mask=pad)
        return h[:, 0, :]

    def embed_query(self, ctx_tokens, pad_mask):
        """Phase 14b/31/32's original path: outfit token as the lead token.
        Kept only as a reference / fallback -- stage 2's actual training and
        eval use embed_query_targeted below, per piece 2 of the brief."""
        h0 = self.embed_set(self.outfit_token, ctx_tokens, pad_mask)
        return F.normalize(self.embed_ffn(h0), p=2, dim=-1)

    def embed_query_targeted(self, ctx_tokens, pad_mask, cat_tokens):
        """Stage 2's real query path: cat_tokens (B, d_model) is the
        target-category token (piece 2), replacing the outfit token as the
        lead position -- exactly the paper's "target item token s" role.
        Output embedding space (d_embed, via embed_ffn) is unchanged."""
        h0 = self.embed_set(cat_tokens, ctx_tokens, pad_mask)
        return F.normalize(self.embed_ffn(h0), p=2, dim=-1)

    def embed_item_alone(self, item_tokens):
        """Unchanged from phase 32: each item as its own length-1 sequence,
        context-independent, precomputable catalog-wide."""
        seq = item_tokens.unsqueeze(1)
        h = self.set_enc(seq, src_key_padding_mask=None)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)

    def cp_score(self, item_tokens, pad_mask):
        """Stage 1 (piece 1): item_tokens is the FULL item set of a
        candidate outfit (real or shuffled-fake), already-projected+
        normalized, no context/target split -- this is a whole-set
        classification, not a retrieval query. Returns raw logits (B,);
        focal loss is applied outside on these."""
        h0 = self.embed_set(self.outfit_token, item_tokens, pad_mask)
        return self.cp_head(h0).squeeze(-1)


def uniformity_loss(f, t=2.0):
    """Wang & Isola (2020) uniformity regularizer, unchanged from phase 14/31/32."""
    sq_dists = torch.cdist(f, f, p=2) ** 2
    B = f.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Standard binary focal loss (Lin et al. 2017), FL(p_t) = -alpha_t (1 -
    p_t)^gamma log(p_t). The paper (section 3.1) names "focal loss" for CP
    pre-training without stating gamma/alpha -- gamma=2.0, alpha=0.25 are
    that paper's own original defaults, adopted here as the standard,
    undocumented-otherwise choice (see implementation_notes.md)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()

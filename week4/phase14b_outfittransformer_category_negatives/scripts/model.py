"""
Phase 14b: architecture reused UNCHANGED from phase 14
(`week4/phase14_outfittransformer_siglip/scripts/model.py`, copied verbatim,
not modified in any way here). Per the professor's redirect brief
(`week4/phase14b_outfittransformer_category_negatives.md`), only the
NEGATIVE SAMPLING changes in this phase -- the set-encoder mechanism, the
frozen-SigLIP item projection, the outfit token, and the loss function
itself (in_batch_triplet_loss, still a triplet-margin comparison of a
positive distance against a hardest-negative distance) are identical to
phase 14. What changes is WHERE the negative embeddings that feed that same
loss function come from: phase 14 used other outfits' targets already
present in the same training batch (no category restriction); phase 14b
draws them from a same-category candidate pool instead (see
../scripts/train_core.py). See phase 14's own docstring below for the full
architecture trace and deliberate-deviation notes -- none of that changed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_DIM = 768
D_MODEL = 128     # item-token dimension, matches the repo's item_enc_dim_per_modality
D_EMBED = 64      # final query/candidate embedding dim, matches phase 13b's CSA-Net-on-SigLIP for direct comparability
N_HEADS = 8
N_LAYERS = 4
D_FFN = 512
DROPOUT = 0.1
MARGIN = 0.3      # rescaled down from the repo's margin=2.0 (see phase 14's module docstring: embeddings are L2-normalized here, so max Euclidean distance is 2, not unbounded -- 0.3 matches this project's own CSA-Net margin, already known to be a sane value on a unit hypersphere)


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
        self.outfit_token = nn.Parameter(torch.randn(d_model) * 0.02)  # matches the repo's own token-init scale

    def encode_item_tokens(self, siglip_vecs):
        """siglip_vecs: (N, 768) raw SigLIP features, any norm. Returns
        (N, d_model) item tokens, L2-normalized -- mirrors the repo's own
        _style_enc_forward normalization step applied before the transformer
        (its non-concat branch: F.normalize(embs_of_inputs, ...))."""
        x = F.normalize(siglip_vecs, p=2, dim=-1)
        e = self.proj(x)
        return F.normalize(e, p=2, dim=-1)

    def embed_query(self, ctx_tokens, pad_mask):
        """ctx_tokens: (B, L, d_model) already-projected+normalized context
        item tokens (from encode_item_tokens). pad_mask: (B, L) bool, True
        marks a padded (non-real) position. Returns (B, d_embed) L2-normalized
        outfit-query embeddings, read from the outfit token's output
        position -- exactly the repo's embed_query."""
        B = ctx_tokens.shape[0]
        tok = self.outfit_token.view(1, 1, -1).expand(B, -1, -1)
        seq = torch.cat([tok, ctx_tokens], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device), pad_mask], dim=1)
        h = self.set_enc(seq, src_key_padding_mask=pad)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)

    def embed_item_alone(self, item_tokens):
        """item_tokens: (N, d_model) already-projected+normalized item
        tokens. Each item is treated as its own length-1 sequence (no outfit
        token) -- exactly the repo's embed_item. Context-independent, so
        this can be precomputed once for an entire catalog."""
        seq = item_tokens.unsqueeze(1)  # (N, 1, d_model)
        h = self.set_enc(seq, src_key_padding_mask=None)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)


def uniformity_loss(f, t=2.0):
    """Wang & Isola (2020) uniformity regularizer, identical to phase 14's
    (and phases 13/13b's before that) -- on from the start of training here
    too, since phase 14's own smoke test already established this mechanism
    needs it to avoid direction collapse."""
    sq_dists = torch.cdist(f, f, p=2) ** 2
    B = f.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

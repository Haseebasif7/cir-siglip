"""
Phase 31: OutfitTransformerSigLIP architecture, copied from
week4/phase14b_outfittransformer_category_negatives/scripts/model.py with NO
structural changes -- every constructor parameter (siglip_dim, d_model,
d_embed, n_heads, n_layers, d_ffn, dropout) was already exposed there, so
step 1 (text input) and step 4 (scale testing) both just instantiate this
class with different arguments; nothing in the class body itself needed to
change. siglip_dim=768 for image-only, 1536 for image+text (base_repr =
normalize(concat(image_768, text_768)), identical construction to phase
27/28/30's own text integration).

Kept as a standalone file (not imported from phase 14b), per this project's
established cross-phase convention of each phase folder being self-contained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_DIM = 768        # 1536 with text input (step 1)
D_MODEL = 128            # item-token dimension -- scale axis (step 4d)
D_EMBED = 64             # final query/candidate embedding dim -- scale axis (step 4e)
N_HEADS = 8              # scale axis (step 4a)
N_LAYERS = 4             # scale axis (step 4b)
D_FFN = 512              # scale axis (step 4c)
DROPOUT = 0.1            # scale axis (step 4f, conditional)
MARGIN = 0.3             # loss-shape axis (step 3b)


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

    def encode_item_tokens(self, siglip_vecs):
        """siglip_vecs: (N, siglip_dim) raw features (or concat(image,text)),
        any norm. Returns (N, d_model) item tokens, L2-normalized."""
        x = F.normalize(siglip_vecs, p=2, dim=-1)
        e = self.proj(x)
        return F.normalize(e, p=2, dim=-1)

    def embed_query(self, ctx_tokens, pad_mask):
        """ctx_tokens: (B, L, d_model) already-projected+normalized context
        item tokens. pad_mask: (B, L) bool, True marks a padded position.
        Returns (B, d_embed) L2-normalized outfit-query embeddings, read from
        the outfit token's output position."""
        B = ctx_tokens.shape[0]
        tok = self.outfit_token.view(1, 1, -1).expand(B, -1, -1)
        seq = torch.cat([tok, ctx_tokens], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device), pad_mask], dim=1)
        h = self.set_enc(seq, src_key_padding_mask=pad)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)

    def embed_item_alone(self, item_tokens):
        """item_tokens: (N, d_model) already-projected+normalized item
        tokens. Each item is its own length-1 sequence (no outfit token) --
        context-independent, precomputable catalog-wide."""
        seq = item_tokens.unsqueeze(1)  # (N, 1, d_model)
        h = self.set_enc(seq, src_key_padding_mask=None)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)


def uniformity_loss(f, t=2.0):
    """Wang & Isola (2020) uniformity regularizer, unchanged from phase 14/14b."""
    sq_dists = torch.cdist(f, f, p=2) ** 2
    B = f.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

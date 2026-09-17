"""
Phase 14: OutfitTransformer's set-encoder mechanism (transformer over an
outfit's items + a learnable "outfit token"), adapted to run on frozen
SigLIP embeddings instead of the paper's own CLIP image+text item encoder.

Architecture confirmed directly from github.com/bigohofone/outfit-transformer's
source (src/models/outfit_transformer.py, src/utils/loss.py,
src/data/datasets/polyvore.py -- see ../architecture_notes.md for the full
trace) -- NOT from the paper's own text alone. Key mechanism, faithfully
reproduced:
  - a per-item embedding (here: L2-normalized SigLIP -> Linear projection,
    replacing the paper's own CLIP-based item encoder)
  - a small transformer encoder (nn.TransformerEncoder, norm_first, mish
    activation, following the repo's own layer config) processes a *set* of
    item tokens
  - for the outfit's query representation: a learnable "outfit token" is
    prepended to the context items' tokens, the whole sequence goes through
    the transformer, and the OUTPUT AT THE TOKEN'S POSITION is the outfit's
    embedding (exactly the repo's `embed_query`)
  - for a candidate item's representation: the item is run through the SAME
    transformer as its own length-1 sequence (no outfit token) -- exactly
    the repo's `embed_item`. This is what makes candidate embeddings
    context-independent and precomputable for the whole catalog once,
    the same trick phase 13b used for CSA-Net's SigLIP variant.

Deliberate deviations from the repo, documented in architecture_notes.md:
  - single visual modality only (this project has never extracted a text
    encoder for these items in any phase); the repo's "concat two 128-dim
    modalities" step collapses to a single unnormalized-then-projected
    branch, matching its own explicit non-concat code path
  - final embeddings ARE L2-normalized before any distance computation
    (repo default: transformer_norm_out=False) -- required by this
    project's own phase 13 lesson that an unnormalized margin/triplet loss
    has a trivial "collapse toward the origin" solution; the loss margin is
    rescaled accordingly (see below)
  - smaller transformer (4 layers / 8 heads / d_ffn=512 vs. the repo's 6
    layers / 16 heads / d_ffn=2024) -- the brief's own step 2 asks for "a
    small transformer encoder", and there is no full CLIP fine-tuning signal
    here to justify the repo's larger capacity
  - no target-item-category conditioning of the outfit token -- confirmed
    the repo's own `embed_query` does not take the target category as input
    either (it only distinguishes the compatibility-prediction task from
    the retrieval task via a fixed additive task embedding, see
    architecture_notes.md), so this is not a simplification introduced
    here, it is faithful to what the actual reference implementation does
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
MARGIN = 0.3      # rescaled down from the repo's margin=2.0 (see module docstring: embeddings are L2-normalized here, so max Euclidean distance is 2, not unbounded -- 0.3 matches this project's own CSA-Net margin, already known to be a sane value on a unit hypersphere)


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


def in_batch_triplet_loss(query_emb, answer_emb, margin=MARGIN):
    """In-batch hardest-negative triplet margin loss on Euclidean distance
    between L2-normalized query and answer embeddings -- exactly the repo's
    InBatchTripletMarginLoss (src/utils/loss.py), with margin rescaled for
    a normalized embedding space (see module docstring). Returns (loss,
    mean positive distance, mean hardest-negative distance) -- the last two
    feed this project's standard D_pos/D_neg ranking diagnostic."""
    dists = torch.cdist(query_emb, answer_emb, p=2)  # (B, B)
    pos = torch.diag(dists)
    neg = dists.clone()
    neg.fill_diagonal_(float("inf"))
    hardest_neg, _ = neg.min(dim=1)
    loss = F.relu(pos - hardest_neg + margin)
    return loss.mean(), pos.mean().item(), hardest_neg.mean().item()


def uniformity_loss(f, t=2.0):
    """Wang & Isola (2020) uniformity regularizer, identical to phases
    13/13b's -- kept ready to add on top of the triplet loss if early
    training shows any sign of direction collapse (see step 3 of the brief
    and ../training_log.md for whether it was actually triggered)."""
    sq_dists = torch.cdist(f, f, p=2) ** 2
    B = f.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=f.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())

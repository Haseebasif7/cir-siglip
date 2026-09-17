# Phase 33, Step 2: Text Input Integration

## The change

`CSANetSigLIP.__init__` already exposed `siglip_dim` as a constructor parameter
(`week4/phase13b_csa_net_siglip_backbone/scripts/model.py:29`), so no architecture code changed at all --
exactly the same situation phase 31 found for OutfitTransformer. `self.proj = nn.Linear(siglip_dim,
embed_dim)`, `siglip_dim` set to 768 (image-only) or 1536 (`normalize(concat(image_768, text_768))`,
identical construction to phase 27/28/31/32). Text embeddings reused directly from
`week7/phase27_text_and_category/data/text_embeddings.npz`, no re-extraction.

The category-pair-conditioned subspace attention mechanism itself is completely unchanged -- text enters
only through `encode_feature`'s input, the same single entry point used for every role (context items,
positive, negatives, eval candidates, eval query context).

## Parameter delta

`self.proj`: 768*64+64=49,216 params (image-only) vs. 1536*64+64=98,368 params (image+text) --
theoretical delta 49,152. Confirmed directly from the trained checkpoints: A2 (image-only) has 51,333
total params, A3 (image+text) has 100,485 -- difference 49,152, exact match.

## Isolated contribution (A3 vs. A2, everything else held fixed)

Both at `lr=5e-5, batch_size=96, patience=5, selection_metric=recall10` (step 1's adopted configuration):

| Configuration | Val Recall@10 | Best epoch |
|---|---|---|
| A2: image only | 0.0779 | 30 |
| **A3: image + text** | **0.0986** | 30 |

**+26.6% relative** -- a real, substantial, isolated gain from text alone, consistent in direction with
every other phase in this project that's tested text (27/28/31/32), though the exact magnitude differs by
architecture, as expected. Both runs ran locally (free) on the M4 Air, ~28 minutes each with the vectorized
training core (`train_core.py`) -- see `../phase33_notes.md` for the full timing comparison against phase
13b's original ~6.9-hour run.

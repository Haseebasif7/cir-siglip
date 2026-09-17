# Phase 29: Context Training Pairs

Built from the same official Polyvore nondisjoint train/valid outfit files phase 9 used, kept in their natural multi-item outfit grouping instead of flattened to pairwise anchor->positive edges. Necessary so candidate-conditioned cross-attention has more than one context item to learn to weight during training -- see architecture_notes.md for why this is required, not optional scope.

- Shared item universe: 251008 items (same as phase 27/28's siglip_base.npz / text_embeddings.npz).
- Train: 284767 context examples from 53306 outfits (0 outfits skipped for <2 valid items, 0 item-slots dropped for missing embeddings).
- Val: 26781 context examples from 5000 outfits (0 outfits skipped, 0 item-slots dropped).
- Context length across all examples: min=1, max=18, mean=4.83 (mirrors the real CIR benchmark's own query_items length distribution, mean 4.86, since both are built from the same underlying outfit structure).
- Output: `data/context_training_pairs.json` (311548 total examples), to be uploaded to the Modal volume alongside phase 27/28's existing data.

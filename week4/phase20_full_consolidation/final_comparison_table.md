# Phase 20: Final Comparison Table, Full Redirected Sequence

Every configuration tested across the redirected sequence (points 1-4) and its immediate predecessors, evaluated on the identical CIR benchmark (`week4/phase12_controllable_modes/data/cir_benchmark.json`, 29,681 queries, 0 skipped, every "mine" row below).

**Labeling convention, kept consistent with every prior comparison in this project:**
- **"mine"** = trained/measured in this project, on this exact harness, frozen SigLIP backbone throughout.
- **"published"** = the paper's own reported number, different backbone (CLIP) and candidate-pool construction, directional context only, not a matched-protocol claim.

## Full table, ordered by Recall@10 ascending

| Configuration | Source | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|---|
| Raw SigLIP (frozen, no training) | mine | 0.0553 | 0.1067 | 0.1437 |
| Phase 19: single-vector feature weighting | mine | 0.0577 | 0.1191 | 0.1616 |
| Phase 14b, run 2: OutfitTransformer, category-restricted random negatives (redirect's current main baseline) | mine | 0.0588 | 0.1286 | 0.1809 |
| Phase 13b: CSA-Net mechanism, frozen SigLIP backbone (the attention candidate) | mine | 0.0725 | 0.1393 | 0.1844 |
| OutfitTransformer published (paper, Polyvore Outfits) | published | 0.0958 | 0.1796 | 0.2198 |
| **Phase 9: plain ProjectionHead, Polyvore-trained (the projection candidate)** | **mine** | **0.1317** | **0.2464** | **0.3216** |

**Phase 9's plain projection mechanism is the strongest configuration in this entire table, at every K, by a wide margin** -- roughly double the next-best "mine" row (CSA-Net's reproduction) and roughly 1.4-1.5x OutfitTransformer's own published numbers, using nothing more architecturally complex than a two-layer MLP (768 -> 256 -> 128) trained with a standard contrastive loss on real outfit co-occurrence, no attention, no conditioning, no dial.

## Every configuration against phase 9 (the strongest result)

| Configuration | Phase 9's ratio to it, K=10 | K=30 | K=50 |
|---|---|---|---|
| Raw SigLIP | 2.38x | 2.31x | 2.24x |
| Phase 19 feature weighting | 2.28x | 2.07x | 1.99x |
| Phase 14b OutfitTransformer reproduction | 2.24x | 1.92x | 1.78x |
| Phase 13b CSA-Net reproduction | 1.82x | 1.77x | 1.74x |
| OutfitTransformer published | 1.37x | 1.37x | 1.46x |

Phase 9 beats every other configuration in this table at every single K -- see `success_bar_check.md` for the explicit, per-K check against the professor's own stated bar.

# Phase 14: OutfitTransformer's Set-Encoder Mechanism on Frozen SigLIP -- Results Table

Evaluated on this project's own CIR harness
(`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged since
phase 12): 29,681 queries, 0 skipped. Trained to genuine convergence (100
epochs, LR decayed to ~0, early-stopping patience reached right at the
schedule's end -- best checkpoint from epoch 91, see `training_log.md`),
**not** budget-truncated the way phase 13's ResNet18 reproduction was.

**Labeling, kept distinct per this project's established convention:**
- **"This configuration"** = OutfitTransformer's set-encoder mechanism
  (transformer + learnable outfit token), trained on top of a FROZEN SigLIP
  backbone (this phase).
- **"OutfitTransformer published"** = the paper's own numbers, their own
  backbone (CLIP) and their own candidate-pool protocol -- not a
  byte-for-byte matched evaluation, a similar-protocol reference only, same
  caveat used throughout this project for every literature-anchor
  comparison.

## This configuration vs. OutfitTransformer's published numbers

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| **This configuration (OutfitTransformer mechanism, frozen SigLIP)** | **0.0201** | **0.0588** | **0.0911** |
| OutfitTransformer published (paper, Polyvore Outfits) | 0.0958 | 0.1796 | 0.2198 |

This configuration reaches **21.0% / 32.7% / 41.4%** of OutfitTransformer's
own published Recall@10/30/50 respectively. Unlike phase 13b's CSA-Net
result (a consistent ~88% across all three K), **this ratio is not flat --
it climbs steadily with K.** See `phase14_notes.md` for why that pattern
itself is diagnostic, not just noise.

## Against this project's own configurations (identical harness, identical benchmark, identical backbone)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| **Phase 14: OutfitTransformer mechanism (this phase)** | **0.0201** | **0.0588** | **0.0911** |
| Raw SigLIP (alone, no training at all) | 0.0553 | 0.1067 | 0.1437 |
| Phase 13b: CSA-Net mechanism (frozen SigLIP) | 0.0725 | 0.1393 | 0.1844 |
| Phase 12c: Substitute mode | 0.0667 | 0.1307 | 0.1734 |
| Phase 12c: Blend (0.5) | 0.0893 | 0.1695 | 0.2255 |
| Phase 12c: Complement mode | 0.0971 | 0.1875 | 0.2471 |
| Phase 9 Model A (alone) | 0.1317 | 0.2464 | 0.3216 |

(Table rows ordered by Recall@10, ascending, so this phase's row's position
relative to every other configuration this project has evaluated on this
exact harness is easy to read at a glance.)

**This configuration is the lowest-scoring row in this entire table, at
every K, including below raw untrained SigLIP.** This is a real,
surprising result -- not what phase 13b's CSA-Net comparison would have
predicted going in, and not explained by embedding collapse (checked
directly, see `training_log.md`: mean pairwise candidate-embedding cosine
similarity is 0.004, i.e. well-spread, not collapsed). See
`phase14_notes.md` for the mechanism this phase believes explains it.

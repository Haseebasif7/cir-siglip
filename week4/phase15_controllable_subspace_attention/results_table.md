# Phase 15: Control-Conditioned Subspace Attention -- Results Table

Evaluated on this project's own CIR harness
(`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged since
phase 12): 29,681 queries, 0 skipped. **Best configuration = the
discrete-alpha-trained checkpoint** (`ablation_results.md`: beats
continuous-alpha training at every alpha, every K, and on the smoothness
diagnostic). Full 11-point alpha sweep in `data/cir_sweep_discrete.json`;
this table reports the alpha=0 (complement-leaning) and alpha=1
(substitute-leaning) endpoints for direct comparability with every other
configuration's own single-number results.

**Labeling, kept consistent with this project's established convention:**
- **"This phase (discrete, alpha=0/1)"** = the control-conditioned CSA-Net
  subspace attention mechanism (this phase's architectural extension),
  discrete-alpha-trained, frozen SigLIP backbone.
- **Published baselines** = the papers' own numbers, their own backbone and
  candidate-pool protocol -- not a byte-for-byte matched evaluation, a
  similar-protocol reference only, same caveat used throughout this project.

## This phase vs. both published literature baselines

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| **This phase, alpha=0 (complement-leaning)** | **0.0657** | **0.1326** | **0.1792** |
| **This phase, alpha=1 (substitute-leaning)** | **0.0634** | **0.1294** | **0.1754** |
| CSA-Net published (paper Table 4, Polyvore Outfits non-disjoint) | 0.0827 | 0.1567 | 0.2091 |
| OutfitTransformer published (paper, Polyvore Outfits) | 0.0958 | 0.1796 | 0.2198 |

This phase's alpha=0 endpoint reaches **79.4% / 84.6% / 85.7%** of CSA-Net's
published Recall@10/30/50, and **68.6% / 73.8% / 81.5%** of
OutfitTransformer's published numbers.

## Against this project's own configurations (identical harness, identical benchmark, identical backbone)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 14: OutfitTransformer mechanism | 0.0201 | 0.0588 | 0.0911 |
| Raw SigLIP (alone, no training) | 0.0553 | 0.1067 | 0.1437 |
| **Phase 15: this phase, alpha=1 (substitute-leaning)** | **0.0634** | **0.1294** | **0.1754** |
| Phase 12c: Substitute mode | 0.0667 | 0.1307 | 0.1734 |
| **Phase 15: this phase, alpha=0 (complement-leaning)** | **0.0657** | **0.1326** | **0.1792** |
| Phase 13b: CSA-Net mechanism (fixed, non-conditional) | 0.0725 | 0.1393 | 0.1844 |
| Phase 12c: Blend (0.5) | 0.0893 | 0.1695 | 0.2255 |
| Phase 12c: Complement mode | 0.0971 | 0.1875 | 0.2471 |
| Phase 9 Model A (alone) | 0.1317 | 0.2464 | 0.3216 |

(Ordered by Recall@10, ascending.)

**This phase's control-conditioned mechanism lands in the middle of this
project's own ranking** -- clearly better than phase 14's OutfitTransformer
mechanism and raw SigLIP, roughly on par with phase 12c's substitute mode,
but **below phase 13b's own fixed (non-conditional) CSA-Net mechanism at
both alpha endpoints**, and well below phase 12c's blend/complement modes
and phase 9's simpler model.

### Endpoint consistency check (pre-declared tolerance, see `training_log.md`)

This phase's alpha=0 endpoint (0.0657/0.1326/0.1792) is **90.6% / 95.2% /
97.2%** of phase 13b's complement-only-trained CSA-Net (0.0725/0.1393/0.1844)
-- well within the pre-declared +-20% tolerance band. Per the plan's
pre-committed rule, this is unsurprising and needs no further explanation
beyond "joint substitute+complement training changed the shared parameters
somewhat, as expected" -- it is NOT a large enough shortfall to invoke phase
12c's documented capacity-competition trend as an explanation.

## Why this configuration doesn't beat phase 13b's fixed mechanism, despite adding conditioning

The added alpha-conditioning capacity did not translate into a Recall@K
improvement over the simpler, non-conditional CSA-Net baseline it extends.
Per `architecture_notes.md`'s and `ablation_results.md`'s full analysis:
`attn_net` did learn a real, measurable response to alpha (attention-weight
shift 0.486 for the discrete checkpoint, far above the untrained baseline of
0.041), but that response is concentrated in a subset of category pairs
rather than distributed evenly, so its effect on AGGREGATE Recall@K is small
relative to phase 12c/12d's simpler additive mechanism's much larger,
smoother controllable range on the identical backbone. See
`phase15_notes.md` for the full interpretation.

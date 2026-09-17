# Phase 16d: Architecture Notes

## The diagnosed problem this architecture change targets

Phase 16c's mode-vector diagnostic showed the fix that separated the two
modes' behavior (attribute-based tail signal) worked exactly as intended
representationally (mode-vector cosine similarity dropped to 0.059, near-
orthogonal), but relevance mode's own validation loss got measurably worse
(0.9735 -> 1.1272, +15.8% relative) than phase 16's, and both axis checks
reversed direction. Phase 16's own mode-vector-magnitude check found the
root structural cause: `ControllableProjectionHead` shares essentially all
its capacity (a full 768->256->128 MLP), with each mode getting only a
small additive 128-d correction vector on top -- 14-16% of the shared
trunk's typical output norm. A strong, structurally different objective
(phase 16c's attribute signal) can pull the shared trunk toward its own
optimum hard enough that the weaker mode-specific correction can't fully
counteract it for the OTHER mode, degrading that mode's quality as a side
effect -- capacity competition.

## The fix: dedicated per-mode capacity, minimal shared trunk

```
frozen SigLIP (768-d)
        |
   Linear(768, 256) -> ReLU -> Dropout(0.1)      <- SHARED, dimensionality
        |                                            reduction only, NOT
        |                                            the full transformation
        +----------------------+
        |                      |
Linear(256, 128)        Linear(256, 128)          <- DEDICATED, one full,
  relevance_head           tail_head                  independently-trained
        |                      |                      linear transformation
   L2-normalize           L2-normalize                per mode
        |                      |
        z_rel                z_tail
              \              /
       alpha * z_rel + (1-alpha) * z_tail
                    |
              L2-normalize (final)
```

**What's shared**: only the first linear layer (768->256), doing
dimensionality reduction, not a learned representation each mode has to
compete over. This is a deliberate, minimal amount of shared structure --
enough that both modes' outputs descend from the same early representation
(keeping them in a broadly compatible coordinate space, which matters for
interpolation to be meaningful at all), but not enough to be a bottleneck
either mode has to win a tug-of-war over.

**What's dedicated**: the full 256->128 transformation for each mode --
substantially more capacity than phase 16/16c's small additive 128-d
vector (which was, in effect, a rank-limited correction on top of an
otherwise fully shared 768->256->128 transformation). Each head is
independently trainable and only ever receives gradient from its own
mode's loss at the alpha extremes used during training (see
`model.py`'s docstring: at `alpha=1.0`, the tail head's contribution to
the blended output is multiplied by exactly `(1-alpha)=0`, so it gets zero
gradient from the relevance loss for that forward pass, and symmetrically
for the tail loss at `alpha=0.0`).

**Inference-time interpolation, unchanged in spirit from every prior phase
in this project**: each head's own output is L2-normalized independently
first, then blended by alpha, then the blend itself is renormalized. This
was a deliberate design decision from the brief, not assumed to behave
sensibly -- the smoothness check (`smoothness_check.md`) is the direct
empirical test of whether blending two independently-normalized,
independently-trained embeddings still produces a coherent, continuously
interpolating dial rather than an incoherent jump between two unrelated
spaces.

## What's reused unchanged

- Phase 16c's attribute-based training pairs (`week5/phase16c_attribute_tail_signal/data/attribute_pairs.json`)
  -- reused directly, no rebuild. That data design (fine-grained-category
  pairs, deliberately tail-stratified) was already proven correct in
  phase 16c; only the architecture is under test this phase.
- Phase 7's full also_buy edge set for relevance mode, exactly as phases
  16 and 16c used it.
- `mnrl_loss` (unweighted, both modes), the negative-sampling convention
  (R_NEG=8 random negatives), the calibration-then-gradient-norm-
  verification loss-balancing procedure, and the collapse-guard/
  uniformity-regularizer safeguards -- all unchanged in method, only now
  applied to the shared layer's parameters specifically (`model.shared`,
  not `model.net`) since that's the only point where the two objectives'
  gradients still interact.

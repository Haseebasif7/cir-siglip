# Phase 16d: Loss Balancing Check (Shared Layer)

Measured fresh on the freshly-initialized (untrained) `DedicatedCapacityHead`, seed=42 -- not assumed from phases 16/16c's weights, per this project's standing rule: both the architecture and the point of interaction between the two objectives changed.

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

- Relevance loss (MNRL/InfoNCE, unweighted, full also_buy set): 4.7836
- Tail-exposure loss (MNRL/InfoNCE, unweighted, attribute-pair set reused from phase 16c): 4.8291
- Ratio: 0.9906x

**Chosen weight_tail = initial_rel / initial_tail = 0.9906**.

## Gradient-norm verification (shared layer only)

Gradient L2-norm into `model.shared`'s parameters (the ONLY shared submodule in this architecture -- unlike phases 16/16c, where the whole 768->256->128 trunk was shared), each loss term isolated, one fixed batch of 128 anchor edges per population:

| Loss term | Grad norm into shared layer | Ratio to relevance |
|---|---|---|
| Relevance (MNRL) | 0.732587 | 1.0x (reference) |
| Tail-exposure, UNWEIGHTED | 0.574157 | 1.28x |
| Tail-exposure, weighted x0.99 | 0.579220 | 1.26x |

**Verification PASSED**: after weighting, the tail-exposure loss's gradient into the shared layer is within a reasonable order of magnitude of the relevance loss's (1.26x). Proceeding to full training with this fixed weight_tail.

# Phase 16c: Loss Balancing Check

Measured fresh on the freshly-initialized (untrained) `ControllableProjectionHead`, seed=42 -- NOT assumed to match phase 16's weight, per this project's standing rule: the loss population changed (attribute pairs instead of IPS-weighted also_buy edges), so recalibrate from scratch.

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

- Relevance loss (MNRL/InfoNCE, unweighted, full also_buy set): 4.7836
- Tail-exposure loss (MNRL/InfoNCE, unweighted, attribute-pair set): 4.8263
- Ratio: 0.9912x

**Chosen weight_tail = initial_rel / initial_tail = 0.9912**.

## Gradient-norm verification

Gradient L2-norm into the shared base projection's parameters (`model.net`), each loss term isolated, one fixed batch of 128 anchor edges per population:

| Loss term | Grad norm into shared net | Ratio to relevance |
|---|---|---|
| Relevance (MNRL) | 0.908113 | 1.0x (reference) |
| Tail-exposure, UNWEIGHTED | 0.794848 | 1.14x |
| Tail-exposure, weighted x0.99 | 0.780909 | 1.16x |

**Verification PASSED**: after weighting, the tail-exposure loss's gradient into the shared net is within a reasonable order of magnitude of the relevance loss's (1.16x). Proceeding to full training with this fixed weight_tail.

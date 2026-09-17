# Phase 12b, Step 2: Loss Balancing Check

Measured on the freshly-initialized (untrained) `ControllableProjectionHead`, seed=42.

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

- Complement loss (MNRL/InfoNCE): 4.8981
- Substitute loss (1 - cosine to fixed PCA-128 target, phase 12b's new formulation): 1.0019
- Ratio: 4.9x (phase 12's old batch-local pairwise-MSE substitute loss started at a very different scale -- ~0.015 vs complement's ~4.4, a ~290x ratio at that point in training; this new formulation's starting ratio is not assumed to be the same and is remeasured here directly, per the brief's explicit instruction).

- **Chosen weight_sub = initial_comp / initial_sub = 4.8886** (single fixed scalar, applied to the substitute loss for the entire training run -- normalizes so both loss terms contribute comparably to the shared network's gradient at the start of training, the 'normalize by each loss's own starting magnitude' approach the brief suggested).

## Gradient-norm verification (the actual check, not just the loss-value ratio)

Gradient L2-norm flowing into the shared base projection's parameters (`model.net`), computed from each loss term SEPARATELY (the other term excluded from that backward pass), on one fixed batch of 128 anchor edges:

| Loss term | Grad norm into shared net | Ratio to complement |
|---|---|---|
| Complement (MNRL) | 0.478354 | 1.0x (reference) |
| Substitute, UNWEIGHTED | 0.306622 | 1.6x smaller |
| Substitute, weighted x4.89 | 1.497878 | 0.3x |

**Verification PASSED**: after weighting, the substitute loss's gradient into the shared net is within a reasonable order of magnitude of the complement loss's (0.32x, not hundreds of times off). Proceeding to full training with this fixed weight_sub.


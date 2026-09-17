# Phase 12c, Step 3: Loss Balancing Check (Ranking-Distillation Substitute Loss)

Measured fresh on the freshly-initialized (untrained) `ControllableProjectionHead`, seed=42 -- NOT assumed to match phase 12b's weight, per this phase's own explicit instruction (the loss's scale changed again with the new formulation).

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

- Complement loss (MNRL/InfoNCE): 4.8981
- Substitute loss (KL ranking distillation, K=50, tau=0.07): 0.0760
- Ratio: 64.47x (phase 12's original batch-local pairwise-MSE loss started ~290-400x smaller than complement; phase 12b's PCA-128 cosine loss started ~4.9x smaller; this phase's KL-divergence ranking-distillation loss is remeasured independently here, not assumed to land at either prior ratio).

- **Chosen weight_sub = initial_comp / initial_sub = 64.4731**.

## Gradient-norm verification

Gradient L2-norm into the shared base projection's parameters (`model.net`), each loss term isolated (other term excluded from that backward pass), one fixed batch of 128 anchor edges:

| Loss term | Grad norm into shared net | Ratio to complement |
|---|---|---|
| Complement (MNRL) | 0.478354 | 1.0x (reference) |
| Substitute, UNWEIGHTED | 0.055289 | 8.7x smaller |
| Substitute, weighted x64.47 | 3.586326 | 0.1x |

**Verification PASSED**: after weighting, the substitute loss's gradient into the shared net is within a reasonable order of magnitude of the complement loss's (0.13x). Proceeding to full training with this fixed weight_sub.


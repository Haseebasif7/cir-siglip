# Phase 17: Loss Balancing Check (Fresh Calibration for Dedicated-Capacity Architecture)

Measured fresh on the freshly-initialized (untrained) `DedicatedCapacityHead`, seed=42 -- NOT assumed to match phase 12c's weight (64.4731x), per the brief's explicit instruction: this is the same loss pair on a structurally different architecture, so the gradient relationship into the shared parameters can differ even if the loss VALUES don't.

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

- Complement loss (MNRL/InfoNCE): 4.8983 (phase 12c: 4.8981)
- Substitute loss (KL ranking distillation, K=50, tau=0.07): 0.0760 (phase 12c: 0.0760)
- Ratio: 64.48x (phase 12c's ratio on `ControllableProjectionHead`: 64.45x). Loss VALUES are architecture-independent at init (both losses only depend on the random projection's outputs, and the two architectures' output distributions at random init are not expected to differ enough to change this ratio much) -- so a similar ratio here is expected and not itself informative; the gradient-norm check below is the real test.

- **Chosen weight_sub = initial_comp / initial_sub = 64.4754**.

## Gradient-norm verification

Gradient L2-norm into the SHARED layer's parameters only (`model.shared`, the 768->256 dimensionality-reduction step -- NOT `model.substitute_head`/`model.complement_head`, which are per-mode and not shared), each loss term isolated, one fixed batch of 128 anchor edges. This is the adapted equivalent of phase 12c's check on `model.net` -- the correct comparison point is the shared parameters specifically, and phase 17's shared layer is much smaller (768->256 only) than phase 12c's full shared trunk (768->256->128), so a different verification outcome here is possible even with an identical loss pair:

| Loss term | Grad norm into shared layer | Ratio to complement |
|---|---|---|
| Complement (MNRL) | 0.462671 | 1.0x (reference) |
| Substitute, UNWEIGHTED | 0.048225 | 9.6x smaller |
| Substitute, weighted x64.48 | 3.198758 | 0.1x |

(Phase 12c's own check, for reference, on `model.net`: weighted ratio landed at 0.1x, weight_sub=64.47.)

**Verification PASSED**: after weighting, the substitute loss's gradient into the shared layer is within a reasonable order of magnitude of the complement loss's (0.14x). Proceeding to full training with this fixed weight_sub.


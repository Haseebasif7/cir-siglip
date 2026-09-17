# Phase 15, Step 3: Loss Balancing Check

Calibrated fresh on the freshly-initialized (untrained) `CSANetSigLIPControllable`, seed=42 -- NOT carried over from phase 12c's weight_sub=64.47, since this is a new combination of loss formulations on a different (CSA-Net) architecture entirely, per this phase's own explicit instruction not to assume a previously-measured ratio applies.

## Reconciliation rule (decided in advance, see architecture_notes.md decision 6)

`weight_sub` is calibrated at alpha=0.5 ONLY -- the one point where the loss envelope (`alpha * weight_sub * substitute + (1-alpha) * complement`) treats both terms symmetrically, so scale-matching (what weight_sub is for) and alpha-driven emphasis (what the envelope itself is for) are cleanly separable. The gradient-norm checks at alpha in {0.2, 0.5, 0.8} below are **verification, not re-calibration**: at each alpha, `weight_sub * substitute_loss`'s gradient norm into the shared params (isolated from `complement_loss`'s, and from the uniformity regularizer, which couples both branches together and would contaminate a naive isolation) is compared against `complement_loss`'s own -- the target is a ratio near 1.0 (within a declared 3x tolerance) AT EVERY alpha checked, meaning weight_sub equalizes the two RAW gradient scales independent of alpha, so the envelope's own alpha/(1-alpha) factor is left to purely control emphasis on top, uncontaminated by a scale mismatch that itself varies with alpha (e.g. via the conditioning input's effect on the forward pass).

## Initial loss magnitude (alpha=0.5, averaged over 20 calibration batches, no optimizer step)

- Complement loss (CSA-Net outfit ranking margin loss): 0.3600
- Substitute loss (KL ranking distillation, same-category neighbors, tau=0.07): 0.0376
- Ratio: 9.58x

**Chosen weight_sub = initial_comp / initial_sub = 9.5801**.

## Gradient-norm verification across alpha in {0.2, 0.5, 0.8}

Gradient L2-norm into the shared parameters (`proj`, `attn_net`, `masks`), each RAW loss term isolated directly (bypassing the uniformity regularizer, which couples both branches via a combined representative-embedding set and would contaminate a naive alpha_weight=0/1 isolation), one fixed 96-outfit batch per alpha, `alpha_forward=alpha` in all cases so the conditioning input's own effect on each term's forward pass is reflected:

| alpha | Complement grad norm | Substitute grad norm (unweighted) | Substitute grad norm (weighted x weight_sub) | Weighted ratio (target ~1.0) | Within tolerance? |
|---|---|---|---|---|---|
| 0.2 | 0.5238 | 0.2049 | 1.9633 | 3.75 | NO |
| 0.5 | 0.6045 | 0.1535 | 1.4705 | 2.43 | YES |
| 0.8 | 0.6038 | 0.1679 | 1.6082 | 2.66 | YES |

**Verification FAILED at: alpha=0.2 (ratio 3.75).** The raw loss-magnitude ratio is itself alpha-dependent (likely batch composition or loss-scale drift across alpha) and a single fixed weight_sub cannot fully correct for it at those points. Proceeding to train with this weight regardless, per the pre-declared fallback -- see training_log.md / phase15_notes.md for how this is accounted for in the final interpretation. An alpha-dependent weight was NOT introduced, since this was declared a fallback only if the check demonstrated it was needed, not assumed upfront.


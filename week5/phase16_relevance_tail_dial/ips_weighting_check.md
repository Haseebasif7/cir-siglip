# Phase 16: IPS Weighting Check

## Data preparation

- Positive edges loaded: `week3/phase7_learned_compatibility/data/positive_edges.json` -- 76293 total (68664 train / 7629 val), the FULL phase 7 also_buy edge set, not phase 8's cross-category-restricted subset.
- Embeddings: `week3/phase7_learned_compatibility/embeddings/siglip_base.npz` (24719 items).
- Disjointness re-assertion against the artifacts this phase actually loads: pool-vs-eval-sample overlap = **0**, training-edge-asins-vs-eval-sample overlap = **0**. Both zero, consistent with `week3/phase7_learned_compatibility/data/exclusion_check.md`.

## Propensity and raw IPS weight

`propensity(item) = (ref_count(item)+1) / (max_ref_count_among_training_targets+1)`, `raw_weight = 1/propensity`, using phase 3's catalog-wide `ref_count` for each edge's TARGET (positive) item.

Max ref_count among the 76293 training targets: **2030**.

Raw weight distribution (before any stabilization):

- min: 1.0000
- median: 48.3571
- 95th percentile: 507.7500
- max: 2031.0000

## Stabilization: percentile capping (not self-normalization)

Justification: batch-level self-normalization (rescaling so a batch's weights sum to a constant) only bounds the batch's weight SUM, not any single example's SHARE of it -- one near-zero-propensity item could still dominate a 128-item batch's gradient almost entirely, which is exactly the instability the phase brief warns about. A fixed global cap directly bounds per-example influence, which addresses that failure mode more directly.

Cap: raw weight clipped at the **95th percentile** (507.7500), computed once over the full 76293-edge training-target distribution (not per-batch -- a 128-item batch's 95th percentile would be a noisy, unstable statistic to cap against).

Fraction of edges with raw weight at or above the cap (i.e. affected by capping): **7.03%**. Given phase 3's own finding that 69.1% of the full catalog has ref_count=0, a large share of training targets landing at or near the cap is EXPECTED here, not a bug -- the tail-exposure mode's entire purpose is elevating that majority. What would actually be a problem is the distribution collapsing to a near-single point mass with no usable variation left; that's checked separately below.

After capping, weights are rescaled to mean 1.0 -- a fixed, one-time affine convenience for interpretability and so this loss's overall scale stays comparable to the unweighted relevance loss going into the gradient-norm calibration (training script, step 3 below). This is NOT a second dynamic stabilization technique on top of capping; it doesn't change which examples get bounded or add any per-batch renormalization.

Post-cap, mean-1-rescaled weight distribution (this is what training actually uses):

- min: 0.0089
- median: 0.4328
- max: 4.5442

## Degeneracy check (stop-before-training gate)

Fraction of weights within 1% of the maximum value: **7.03%** (threshold for 'degenerate, stop' per the brief: >= 98%).

**VERDICT: not degenerate.** There is real, usable variation in the weight distribution beyond the capped ceiling -- proceeding to training.

## Train/eval propensity independence

These capped training weights are a training-loss artifact only. The field-standard evaluation metrics computed later (APRI, RPI, Tail-Coverage@N) use phase 3's raw `ref_count`/`tier` values directly per the brief's own formulas, with no dependency on these capped training weights -- so there is no train/eval propensity mismatch to reconcile.

## Positive sets for in-batch false-negative masking

Built from the full (train+val, both directions) edge set: **14531** anchors have at least one known positive, saved to `positive_sets.json`.

*(Loss-balancing calibration and gradient-norm verification are appended to this file by `03_train.py`.)*

## Loss balancing calibration (03_train.py, phase 12c's two-step procedure)

Measured fresh on the freshly-initialized (untrained) `ControllableProjectionHead`, seed=42 -- not assumed from any prior phase's ratio, per this project's standing rule to recalibrate whenever the loss pair changes.

### Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

- Relevance loss (MNRL/InfoNCE, unweighted): 4.7818
- Tail-exposure loss (MNRL/InfoNCE, IPS-weighted): 4.8431
- Ratio: 0.9873x

**Chosen weight_tail = initial_rel / initial_tail = 0.9873**.

### Gradient-norm verification

Gradient L2-norm into the shared base projection's parameters (`model.net`), each loss term isolated (other term excluded from that backward pass), one fixed batch of 128 anchor edges:

| Loss term | Grad norm into shared net | Ratio to relevance |
|---|---|---|
| Relevance (MNRL) | 0.908699 | 1.0x (reference) |
| Tail-exposure, UNWEIGHTED | 1.019912 | 0.89x |
| Tail-exposure, weighted x0.99 | 1.031596 | 0.88x |

**Verification PASSED**: after weighting, the tail-exposure loss's gradient into the shared net is within a reasonable order of magnitude of the relevance loss's (0.88x). Proceeding to full training with this fixed weight_tail.

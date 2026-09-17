# Phase 18b, Step 1: Contrastive Sub-Tail Signal -- Construction Summary

Positive side: each anchor's top-3 nearest TAIL-TIER neighbors (reused directly from phase 18's `tail_nn_lookup.npz`, itself unchanged raw-SigLIP-derived structure -- not recomputed). Negative side (the actual fix, built at training time, not here): for each anchor, half the negatives are drawn specifically from HEAD-tier items, half uniformly random -- mirroring comp_tail's proven MNRL-with-real-negatives approach, but deliberately biased toward the exact population (popular items) the loss needs to learn to push away from.

- **Positive pairs built: 74157** (66742 train / 7415 val, 10% val split, same convention as phases 7/16c). 0 self-pairs skipped (an anchor that is itself tail-tier, appearing in its own top-3 -- possible only in principle since self is always excluded from the underlying NN lookup already; confirmed 0 in practice below).
- Target-side tail-tier fraction: 1.0000 (expected ~1.0 -- confirms every positive target really is tail-tier, inherited directly from the restricted NN lookup's own column mask, already verified in phase 18).
- **Head-tier items available for negative sampling: 9961/24719 (40.3%)** -- comfortably enough to guarantee real head-tier negatives in every training batch (a batch of 128 anchors x 4 head-biased negatives each needs up to 512 draws, a small fraction of the 9961 available, with replacement across anchors as this project's existing negative-sampling convention already allows).

## Verdict

Coverage is adequate on both the positive and negative sides. Proceeding to step 2 (loss rebalancing) and training -- no reduced K or supplementary data needed.


# Phase 13b: CSA-Net's Subspace Attention on Frozen SigLIP -- Verdict

## What this phase set out to determine

Two things phase 13 couldn't answer because its full reproduction was cut
short by a real budget constraint: (1) whether this project's CIR harness
reproduces something close to CSA-Net's published numbers when the model
actually finishes training, and (2) how this project's own controllable
mechanism (phase 12c) compares against CSA-Net's conditioning mechanism with
the backbone held constant on both sides -- the actually meaningful
comparison for this project's research argument.

## What actually happened

Reusing phase 13's architecture, loss, and fixes exactly (see
`architecture_notes.md`), but swapping the fine-tuned ResNet18 backbone for
a frozen SigLIP embedding plus a small trainable projection, training became
essentially free: no GPU needed, ~7 hours on a local M4 Air, and it ran the
full 40-epoch schedule to genuine convergence (LR decayed to exactly 0,
early stopping never needed -- see `training_log.md`). None of phase 13's
four real optimization failures (OOM, dead gradients, magnitude collapse,
direction collapse) resurfaced, verified directly rather than assumed.

Results (`results_table.md`):

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| This configuration | 0.0725 | 0.1393 | 0.1844 |
| CSA-Net published | 0.0827 | 0.1567 | 0.2091 |
| Phase 13 (undertrained) | 0.0284 | 0.0667 | 0.0960 |
| Raw SigLIP (this project) | 0.0553 | 0.1067 | 0.1437 |
| Phase 9 Model A | 0.1317 | 0.2464 | 0.3216 |
| Phase 12c Substitute | 0.0667 | 0.1307 | 0.1734 |
| Phase 12c Blend (0.5) | 0.0893 | 0.1695 | 0.2255 |
| Phase 12c Complement | 0.0971 | 0.1875 | 0.2471 |

## Step 5: interpret honestly

**Does this land closer to CSA-Net's published numbers than phase 13's
undertrained attempt?** Clearly yes, and by a wide margin -- 0.0725 vs
0.0284 at Recall@10 (2.6x higher), and the gap to the published number
shrinks from "less than half" (phase 13: 34% of published R@10) to "close"
(this phase: 88% of published R@10, and a consistent ~88% across all three
K values). That consistency across K is itself informative: a harness bug
would more plausibly produce an uneven gap (e.g., fine at low K, wrong at
high K, or vice versa) than a uniform ~12% shortfall at every K. This is
reasonable, if not conclusive, evidence that **the CIR harness itself is
sound** -- a genuinely trained model on this harness lands in the right
neighborhood of the published baseline, using a different backbone and a
different (though similarly-constructed) candidate pool. This is the
validation phase 13 couldn't provide because its run never finished. It is
still not a byte-for-byte matched comparison (different backbone than the
paper, different fine-grained-vs-broad category pool construction -- see
`results_table.md`'s labeling notes), so "the harness is probably fine" is
the right-sized claim, not "the harness is proven correct."

**How does this compare to this project's own controllable mechanism
(phase 12c), backbone held constant -- the actual point of this phase?**
Mixed, and worth stating plainly rather than picking the flattering half:

- CSA-Net's mechanism (0.0725/0.1393/0.1844) **beats** phase 12c's
  substitute mode (0.0667/0.1307/0.1734) at all three K values, by a modest
  but consistent margin.
- CSA-Net's mechanism **loses** to phase 12c's blend mode
  (0.0893/0.1695/0.2255) and to phase 12c's complement mode
  (0.0971/0.1875/0.2471) at all three K values, also by a consistent margin.
- Both **lose** to phase 9's plain compatibility model (0.1317/0.2464/0.3216),
  which has no controllable/conditioning mechanism at all -- worth keeping
  in view so neither "mechanism" gets over-credited relative to a simpler
  baseline that just wins outright on this metric.

So: **this project's own mechanism is not uniformly better or worse than
CSA-Net's conditioning mechanism** -- it depends on which of phase 12c's
three operating points (substitute/blend/complement) is the comparison.
Phase 12c's complement mode, specifically, outperforms CSA-Net's mechanism
here. Given phase 12c's controllable mechanism is a genuine dial (phase 12d)
while CSA-Net's is a fixed configuration, a fairer framing is: **CSA-Net's
single fixed operating point sits between this project's substitute and
blend points on Recall@K, closer to substitute** -- not a clean win or loss
for either architecture.

**One honest caveat about THIS run specifically** (see `training_log.md`):
the model converged, but its internal ranking diagnostic (D_pos vs D_neg,
the average distance from the true positive vs. the hardest mined negative)
plateaued after roughly epoch 15 and never reached the point where the true
positive is closer than the hardest negative on average (final gap: +0.082,
wrong sign throughout training). The Recall@K numbers above are still real
and were computed the same way as every other configuration in this
project, so the comparison stands -- but this suggests there may be
headroom left in the mechanism (more negatives, a different aggregation, a
larger attention sub-network) that this specific run's hyperparameters
didn't reach, not that this is necessarily CSA-Net's ceiling on this
backbone.

## Required-output checklist

- `architecture_notes.md` -- done: exact diff from phase 13, what stayed the same.
- `training_log.md` -- done: step-2 verification with evidence, the full
  training run, and the honest plateau observation.
- `results_table.md` -- done: all three CSA-Net-related numbers clearly
  labeled and distinguished, plus every one of this project's own
  configurations.
- This file -- verdict: **the harness is reasonably validated (consistent
  ~88% of published Recall@K across all three K values, using a genuinely
  converged model for the first time)**; **CSA-Net's conditioning mechanism
  does not clearly beat or lose to this project's own controllable mechanism
  on this harness** -- it beats phase 12c's substitute mode, loses to blend
  and complement, and both lose to phase 9's simpler model. No claim of
  "beats CSA-Net" or "CSA-Net beats us" should be made from this phase.

## Do not do yet (per the brief)

Resuming or finishing phase 13's full ResNet18 reproduction remains parked
until GPU budget is available. Starting an OutfitTransformer reproduction
was explicitly deferred until after seeing this phase's result -- given the
mixed, not-clearly-favorable comparison above, that decision should weigh
whether closing CSA-Net's D_pos/D_neg plateau (more negatives, longer
attention sub-network, different aggregation) is a higher-value next step
than a third baseline reproduction. Left open for a fresh conversation.

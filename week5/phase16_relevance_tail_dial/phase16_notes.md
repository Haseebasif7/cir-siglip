# Phase 16: Controllable Relevance / Tail-Exposure Dial -- Notes

## What this phase set out to do

Build a controllable dial, steerable at inference time between accurate/relevant
retrieval and long-tail-favoring retrieval, on Amazon data, reusing phase 12c's
verified `ControllableProjectionHead` architecture, phase 7-9's MNRL/InfoNCE
contrastive loss, and phase 3's catalog-wide popularity/tier lookup for inverse
propensity scoring (IPS). Full design decisions and validated reasoning are in
`/Users/haseeb/.claude/plans/virtual-nibbling-curry.md` (the approved plan for
this phase); this file is the honest verdict once everything ran.

## What actually happened, step by step

**Candidate gallery** (`training_pool_summary.md`): merging phase 7's 24,719-item
pool with phase 1b's 1,872-item eval sample produced a 26,591-item gallery that
turned out to be 43.22% tail-tier (11,493 tail items) -- well above the adequacy
threshold, so no supplementary tail sampling was needed. This is notably better
tail representation than phase 1b's sample alone (0.0% tail, the problem this
gallery merge was built to fix) -- phase 7's larger BFS walk apparently reaches
much further into the tail than phase 1b's smaller one did.

**IPS weighting** (`ips_weighting_check.md`): raw weights ranged from 1.0 to
2031.0 (median 48.4), capped at the 95th percentile (507.75, affecting 7.03% of
edges) and rescaled to mean 1.0. Not degenerate (7.03% of mass at the cap, far
below the 98% stop threshold) -- real, usable variation survived stabilization.

**Training** (`models/training_curves.json`, `logs/step03_train_run.log`): clean
convergence, 42 epochs, early-stopped at epoch 41 (best val_relevance_loss=0.9735
at epoch 36). No collapse at any point -- mean pairwise cosine stayed in the
0.03-0.09 range throughout for both modes, never trending toward the 1.0
collapse ceiling, so the uniformity regularizer (ported in but off by default)
was correctly left disabled; the smoke test's recommendation held for the full
run. Loss balancing calibrated fresh (weight_tail=0.9873, i.e. the two losses
started at nearly identical scale) and the gradient-norm verification passed
comfortably (0.88x ratio, well inside the [0.1, 10] band) -- no scale-mismatch
problem anywhere in this phase.

**Alpha sweep, axis checks, field-standard metrics, smoothness** (full numbers
in `results_table.md` and `smoothness_check.md`): this is where the honest
picture gets more nuanced than "it worked" or "it didn't."

## The honest verdict: a real dial, but a structurally weak one

Every diagnostic that has a direction to check points the **same, correct**
way -- and every one of them is **small**:

- **Axis 1 (relevance mode's Hit Rate@K should exceed tail mode's): FAILS
  the brief's own bar.** Hit Rate@5/@10 are nearly flat across the entire
  sweep (0.3173-0.3194 @5, 0.3937-0.3958 @10) -- alpha=1.0 does not clearly
  beat alpha=0.0 here; the difference is within what looks like noise.
- **Axis 2 (tail mode's mean retrieved ref_count should be lower than
  relevance mode's): PASSES, but modestly.** Mean ref_count@5 rises from
  16.90 (tail) to 17.49 (relevance), @10 from 16.58 to 17.21 -- correctly
  directioned and monotonic-ish across the sweep, but a small swing.
- **APRI (field-standard, `results_table.md`): same story as axis 2** --
  rises from 16.58 (alpha=0) to 17.21 (alpha=1) at N=10, 16.23 to 16.92 at
  N=20. Real, correctly directioned, small.
- **RPI and Coverage@N/Tail-Coverage@N: essentially flat** across the whole
  sweep (RPI ~0.52 everywhere; Coverage@N/Tail-Coverage@N move in the third
  decimal place). No meaningful movement on these axes.
- **Smoothness check (`smoothness_check.md`): monotonic decay confirmed**
  (adjacent-step overlap 0.9931 down to most-distant 0.9396, gap=0.0534),
  so this is a genuine, continuously-interpolating dial, not a discontinuous
  jump -- but the gap is **roughly 9x smaller than phase 12c/12d's own
  reference mechanism** (gap=0.4751 there). Per this project's own
  pre-declared interpretation bands, this lands in "partially confirms,"
  not "confirms": real movement exists, but it's structurally weak.

**Taken together: this phase built a real, working, smoothly-interpolating,
correctly-directioned controllable mechanism -- it is not broken, not
inert, and not a bug -- but it is a substantially weaker dial than phase
12c/12d's substitute/complement mechanism, weak enough that the brief's own
Hit-Rate axis check does not clearly pass.** This is reported plainly rather
than rounded up into a clean "it works" or explained away.

## Why it's weak: a mechanistic diagnosis, not a guess

Two real, checked reasons, not mutually exclusive:

**1. Both modes train on the same task with the same targets -- only the
per-example weighting differs.** Unlike phase 12c, where substitute mode
(ranking-distillation to raw SigLIP neighbors) and complement mode (outfit
co-occurrence) had genuinely *different* training objectives predicting
*different* things, phase 16's relevance and tail-exposure modes are both
trained to predict the exact same also_buy edges from the exact same
positive/negative structure -- IPS reweighting only changes which examples
get more or less emphasis, it never changes what's being predicted. That is
a fundamentally weaker source of push-apart between the two mode vectors
than genuinely different objectives would be, and the evidence is directly
consistent with this: `mode_relevance` and `mode_tail`'s learned norms are
small relative to the shared base projection (0.191 and 0.166 vs. a mean
base-projection norm of 1.233, roughly 14-16%), and the two mode vectors
point in a broadly similar direction (cosine similarity 0.55, not opposed)
rather than diverging sharply. Checked directly (not inferred) via:
```
mode_relevance norm: 0.1914   mode_tail norm: 0.1660
cosine(mode_relevance, mode_tail): 0.5525
base projection mean norm: 1.2334 (std 0.3118)
```

**2. A genuine, previously-documented data-quality property of this
catalog compounds the effect for a real subset of queries.** Two of six
qualitative examples (`qualitative_examples/example_B00265CKZU.png`,
`example_B00794VGQM.png`) turned out to be non-product boilerplate images
(a costume-brand size chart, a UK fit guide) whose top-3 to top-5 nearest
neighbors are near-identical boilerplate images from the same template
family, sitting at raw similarity 0.79-1.000 -- a ceiling the mode vectors'
small perturbation cannot overcome, so those slots stay identical across
the whole alpha sweep regardless of mode. This is not a new problem this
phase introduced: phase 2 and phase 6 already documented this catalog's
non-product-image and near-duplicate-image artifacts. It's a real,
compounding factor for the queries it affects, not the sole explanation for
the aggregate weak effect (a third, cleaner example --
`example_B0160HYB8S.png`, a real umbrella query with real umbrella
retrievals -- shows the mechanism doing something: rank 4/5 genuinely swap
between a head item (ref_count=5) and a tail item (ref_count=0) across the
sweep, though even there the reshuffling is small, confined to the bottom
of the top-5, and not perfectly monotonic at the single-query level --
consistent with a real but modest population-level effect that individual
queries realize noisily).

Both explanations point the same direction: the mechanism has real,
correctly-directioned signal, but not enough separating force to produce a
strong dial the way phase 12c/12d's differently-targeted two-objective
design did.

## Contextual comparison to GUME

Reported as context only, per this project's standing honesty convention --
not a matched-protocol baseline claim. GUME's own published Recall@10/20 on
the same Clothing/Shoes/Jewelry category (0.0703/0.1024) is not directly
comparable to this phase's Hit-Rate@K numbers: different task framing
(GUME predicts held-out user-item interactions on a graph; this phase
retrieves also_buy item-to-item relations against a different, smaller
candidate pool), different metric, different candidate pool size. No claim
of this shape should be made from this phase's results, and none is made.

## What this does and doesn't establish

**Does establish**: an item-only, no-user-history controllable long-tail
mechanism can be built and trained cleanly (no collapse, no loss-balancing
problem, a stabilized and non-degenerate IPS weighting scheme) reusing this
project's existing infrastructure end to end, and it produces real,
correctly-directioned, smoothly-interpolating movement on every diagnostic
that has a direction to check.

**Does not establish**: that this specific mechanism (same-objective,
IPS-reweighted-only differentiation) produces a *strong* or *clearly
useful* dial -- the Hit-Rate axis check fails the brief's own bar, and
every other real effect is small. This is a genuine, honestly-reported
negative/mixed finding with a specific, checked mechanistic explanation,
not a vague "needs more tuning."

## What's left (flagged, not attempted -- out of this phase's scope)

The brief scoped this phase as build-evaluate-report, not iterate-until-
strong (unlike phase 13c/15b's explicit bounded-iteration pattern), so no
further variant was attempted here. If this thread is picked up again, the
mechanistic diagnosis above points at a concrete, specific next step worth
trying: giving the two modes more reason to diverge than reweighting alone
provides, e.g. an explicit repulsion term between `mode_relevance` and
`mode_tail`, or restructuring the tail-exposure objective around a
genuinely different prediction target (e.g. distilling toward tail-item
co-occurrence structure specifically, mirroring how phase 12c's substitute
mode targeted a different structure than its complement mode) rather than
just reweighting the same also_buy targets.

## Documentation

`PROJECT_MEMORY.md` updated with this phase's status per the standing
documentation rule.

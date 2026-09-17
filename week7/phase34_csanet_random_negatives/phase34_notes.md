# Phase 34: Negative-Sampling Fix, the Final Experimental Result

## The central finding, stated plainly

Phase 33 identified CSA-Net's still-mined negative sampling as the single most promising, untested
explanation for why its investment-parity result (0.1247/0.2164/0.2748) trailed OutfitTransformer's own
investment-parity result (0.1897/0.3246/0.4019) by 31.6-34.3% relative, despite receiving the same class of
investment. This phase changed exactly that one variable -- nothing else -- and the result is large and
unambiguous:

**Phase 34's 3-seed CSA-Net ensemble, random same-category negatives, test benchmark:
Recall@10/30/50 = 0.1674/0.2860/0.3586** -- a **+34.2%/+32.2%/+30.5%** relative gain over phase 33's own
investment-parity CSA-Net (mined negatives), and a **+130.9%/+105.3%/+94.5%** relative gain over phase 13b's
original un-invested reproduction.

This closes MOST, but not quite all, of the remaining gap to OutfitTransformer: phase 34 now trails phase
32's investment-parity OutfitTransformer result by **-11.8%/-11.9%/-10.8%** relative (down from phase 33's
-34.3%/-33.3%/-31.6%), and reaches **87.9%/87.5%/87.9%** of phase 28's own 10-model ensemble (up from phase
33's 65.5%/66.2%/67.4%).

## Which of the three outcomes this is, precisely

The phase's own brief named three possible outcomes in advance. The honest answer here is the middle one,
landing close to but not inside the boundary of the first:

- **"Closes substantially"** was defined as landing within roughly 10% relative of OutfitTransformer's
  investment-parity result. Phase 34 lands at **10.8-11.9% away** -- just outside that band, not inside it.
  This is a genuine boundary case and is reported as such, not rounded into either verdict.
- **"Partially closes"** is the outcome that actually applies: a real, large, meaningful improvement over
  phase 33 (+30-34% relative), driven exactly as predicted by this project's own established evidence, but
  still short of full parity with OutfitTransformer's own investment-parity number.
- **"Doesn't move"** clearly does not apply -- this is the largest single-step gain of any phase in this
  project's cross-architecture comparison work.

Stated as precisely as the numbers allow: **negative sampling explains almost the entire remaining gap
identified in phase 33, but not literally all of it.** The residual ~11% gap to OutfitTransformer is now
the honest, final open question for this comparison thread -- and per this phase's own explicit scope, it
is not investigated further here (see "Closing the thread" below).

## Step-by-step progression

| Step | Val R@10 | Relative gain |
|---|---|---|
| Phase 33's own winning config, mined negatives, seed 42 (starting point) | 0.1075 | -- |
| Step 2: same config, seed 42, random negatives | 0.1610 | +49.7% |
| Step 3: 3-seed ensemble (42, 1, 2), random negatives | 0.1805 | +67.9% (vs. the 0.1075 starting point); +11.5% (ensembling alone, over best solo seed 2's 0.1618) |

| Configuration | Test R@10 | Test R@30 | Test R@50 |
|---|---|---|---|
| Phase 13b original (un-invested) | 0.0725 | 0.1393 | 0.1844 |
| Phase 33, investment-parity CSA-Net (mined, 3-seed) | 0.1247 | 0.2164 | 0.2748 |
| **Phase 34, investment-parity CSA-Net (random, 3-seed)** | **0.1674** | **0.2860** | **0.3586** |
| Phase 32, investment-parity OutfitTransformer (random, 3-seed) | 0.1897 | 0.3246 | 0.4019 |
| Phase 28 ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 |

## A second, unprompted piece of corroborating evidence: seed-to-seed consistency

Beyond the mean R@10 gain, random negatives also produced a much tighter cross-seed spread: this phase's
3 seeds have std **0.00087** (`individual_seeds.md`), close to OutfitTransformer's own **0.00074**
(phase 32) and roughly a third of phase 33's mined-negative CSA-Net spread (**0.0027**). Correspondingly,
the ensembling gain alone was smaller here (+11.5% vs. phase 33's +22.7%) -- consistent with this project's
own established pattern that ensembling gains scale with how much individual seeds disagree
(`validation_ensemble_result.md`). This was not something this phase set out to measure, but it points the
same direction as the headline result: mined hard negatives were not just producing a lower mean for
CSA-Net, they were producing a noisier, less consistent optimization target across random seeds.

## What this confirms about the project's broader negative-sampling finding

This project has now confirmed, in a fifth independent context (after phases 7-9, phase 13c, and phase
14b's own run 1 vs. run 2 comparison for OutfitTransformer, with phase 33 providing indirect motivating
evidence), that mined hard negatives underperform random same-category negatives for these embedding
losses -- and for the first time, directly and quantitatively for CSA-Net's own specific architecture and
loss combination, not by analogy from a different architecture. This is a genuinely strong, now
well-replicated finding for this project's methodology section.

## What this means for the cross-architecture comparison, and for the investment-asymmetry claim

Phase 33's own framing asked: does "investment asymmetry explains the apparent architecture gap" generalize
across architectures, or is it specific to OutfitTransformer? This phase sharpens that answer considerably.
**Once the negative-sampling asymmetry between the two architectures' investment histories is corrected,
CSA-Net closes to within about 11-12% of OutfitTransformer's own investment-parity result** (down from
phase 33's 32-34% gap) -- strong evidence that negative-sampling investment was the single largest
component of the architecture-specific gap phase 33 identified, not one of three roughly-equal candidates.

The honest, final statement for this project's central methodological claim: **investment asymmetry
(now understood to include negative-sampling scheme as its single largest component for CSA-Net) explains
the large majority of the apparent architecture gap in this research area, for both architectures tested.**
A residual gap of roughly 11-12% remains between CSA-Net's and OutfitTransformer's investment-parity
results -- smaller than either architecture's own un-invested-to-invested improvement, but real and not
explained by this phase. Phase 33's other two named candidates (ensemble-size distance from ceiling, raw
capacity) remain the most plausible, still-untested explanations for this specific residual -- unchanged
from phase 33's own framing, since this phase deliberately tested only one variable.

## Closing the experimental thread

Per this phase's own explicit brief: **no further experiments are proposed, regardless of outcome, and none
are planned.** The experimental phase of this project's cross-architecture comparison work is complete as
of this phase. The remaining ~11-12% residual gap to OutfitTransformer, and the three originally-named
candidate explanations from phase 33 (of which negative sampling -- now tested -- was the largest), are
reported here as the honest, final boundary conditions on this project's central claim, not as an invitation
for a phase 35. The next step for this project is writing the technical report, not more experiments.

## The honest, final comparison numbers for this project going forward

**Phase 34's 3-seed CSA-Net ensemble (random negatives) -- test-benchmark Recall@10/30/50 =
0.1674/0.2860/0.3586, checkpoints `week7/phase34_csanet_random_negatives/models/csanet34_seed{42,1,2}.pt`
(score-averaged over distances, never embeddings) -- is now the honest CSA-Net baseline citation for this
project, superseding phase 33's own number (0.1247/0.2164/0.2748), which in turn superseded phase 13b's
original (0.0725/0.1393/0.1844). Both earlier numbers remain valid and citable as their own respective
starting points (un-invested, and investment-parity-with-mined-negatives).**

**Phase 28's mean-pooled text-only ensemble (0.1904/0.3267/0.4079) remains this project's best overall
result**, narrowly ahead of both investment-parity baselines: OutfitTransformer (phase 32, under 2%
relative) and CSA-Net (this phase, ~12-13% relative, down from phase 33's 33-35%). Any report language
should state the OutfitTransformer and CSA-Net comparisons separately, per phase 33's own established rule
-- they now tell much more similar, but still not identical, stories, and collapsing them into one number
would still lose the genuine residual difference this phase surfaced.

## For the paper's central argument

This result strengthens the paper's central argument considerably, without overstating it. The pattern from
phase 31/32 (broken investment produces an artificially large apparent gap; fair investment closes it)
now demonstrably extends to a second, mechanistically different architecture, once that architecture's own
full investment history (not just the four dimensions phase 33 tested) is accounted for. The residual
~11-12% gap is reported honestly as an open question with two specific, named, plausible candidates --
exactly the kind of bounded, evidence-backed nuance a discussion section benefits from, rather than either
an overclaimed "fully explained" or an unexplained "still doesn't replicate."

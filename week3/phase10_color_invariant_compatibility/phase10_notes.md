# Phase 10: Color-Invariant Compatibility Training — Notes

## Context

Phase 9 found that its Polyvore-trained compatibility projection transferred worse to Amazon
than even phase 8's own noisy-data-trained model, and identified a likely mechanism
qualitatively: the model appears to lean on color-coordination as a dominant proxy for
"compatibility" — a real, useful pattern within Polyvore's styled photography, but one that
misfires on Amazon's product catalog (the pink laundry bag query retrieving bright-pink items
almost regardless of category). This phase tests the specific, falsifiable hypothesis that
follows directly from that diagnosis: if color-coordination is the shortcut hurting transfer,
explicitly training the model to be invariant to color changes should reduce that reliance and
improve Amazon transfer, likely at some cost to in-domain Polyvore performance.

## Step 1: color perturbation — clean, no issues

Generated a color-perturbed twin for all 220,455 Polyvore train/val items (union of
`positive_edges.json`'s train+val source/target ids) via a deterministic HSV hue shift, drawn
per-item from [40, 320] degrees (excludes a ±40° window around 0 to guarantee a visually
meaningful change), saturation and value untouched. Ran locally (8-way multiprocessing,
~4,500 img/s, under a minute total) — **zero errors, zero missing sources**. Visual
spot-check (`data/color_perturb_summary.md` has the aggregate stats; a few example pairs were
inspected directly during development) confirmed shape/texture/lighting are preserved and only
color identity changes — no need to invoke the "stop and report" fallback in the brief, image
quality was never in question.

## Step 2-3: invariance signal and training — converged cleanly, no collapse

SigLIP extraction for the 220,455 perturbed images moved straight to a Modal L4 GPU (same
workflow as phase 9: tar → Modal Volume → extract to local container disk → batched inference),
~28.5 minutes end to end. Training reused phase 9's exact positive edges (1,373,702 train /
129,850 val, directed co-outfit pairs) and Model A's negative-sampling setup (R=8 random
negatives, no hard negatives, per the brief's instruction to keep this experiment isolated from
the separate hard-negative question). The new invariance loss (1 − mean cosine similarity
between an item's projection from its original vs. perturbed-twin embedding) was added with
equal fixed weighting to the existing MNRL compatibility loss, computed per batch over every
unique item appearing as an anchor or positive.

Training ran fast (~2.5 min/epoch, much quicker than phase 9's benchmarked 6-7 min/epoch on this
hardware — likely system-load-dependent, not a methodology difference) and converged with the
**best checkpoint again at epoch 0** (val loss rose every epoch after), consistent with phase
9's finding that the true optimum sits within epoch 0 at this batch-count scale. No embedding
collapse (`mean_pairwise_cosine` stayed in the 0.67-0.71 range across training, never drifting
toward 1.0).

## Step 4.1: official Polyvore benchmark — a small, expected in-domain cost

| Configuration | Compatibility AUC | FITB Accuracy |
|---|---|---|
| Raw SigLIP | 0.7170 | 0.4840 |
| Phase 9 Model A (naive, no invariance) | 0.9470 | 0.7030 |
| **Phase 10 color-invariant model** | **0.9420** | **0.6907** |

A small drop from phase 9 Model A (AUC −0.005, FITB −0.012), both models still dramatically
above raw SigLIP. This is exactly the shape of result the brief anticipated: giving up a
real, useful in-domain shortcut costs a little accuracy on Polyvore's own domain, where
color-coordination genuinely correlates with real outfit assembly.

## Step 4.2: Amazon transfer test — the central test, and it does NOT clearly support the hypothesis

| Configuration | Full HR@5 | Cross-type HR@5 |
|---|---|---|
| Raw SigLIP | 0.502 | 0.076 |
| Phase 8 Model A (Amazon-trained) | 0.441 | 0.078 |
| Phase 9 Model A (Polyvore-trained, naive) | 0.315 | 0.046 |
| **Phase 10 color-invariant model** | **0.301** | **0.047** |

Full-ground-truth Hit Rate@5 is *slightly worse* than phase 9's naive model (0.301 vs. 0.315),
and cross-type-only Hit Rate@5 is *statistically indistinguishable* (0.047 vs. 0.046 — a
one-query difference at this N). Hit Rate@10 is flat-to-slightly-worse on both views too (full
table in `results_table.md`). **The invariance training did not improve Amazon transfer** —
it paid the step-4.1 in-domain cost without buying back any of the transfer performance the
hypothesis predicted.

## Step 4.3: color-reliance diagnostic — confirms the null result quantitatively

| Configuration | Polyvore r | Amazon r | Combined r |
|---|---|---|---|
| Raw SigLIP | 0.053 | 0.391 | 0.317 |
| Phase 8 Model A (Amazon-trained) | -0.039 | 0.219 | 0.127 |
| Phase 9 Model A (Polyvore-trained, naive) | 0.181 | 0.198 | 0.153 |
| Phase 10 color-invariant model | 0.180 | 0.192 | 0.151 |

(Pearson correlation between each model's predicted compatibility score and raw HSV
histogram-intersection color similarity; 3,000 Polyvore test-outfit pairs + 6,605 Amazon
also_buy pairs.) **The correlation barely moved**: 0.181→0.180 on Polyvore, 0.198→0.192 on
Amazon — differences an order of magnitude smaller than what would be needed to call this a
meaningful reduction in color reliance. This directly confirms, quantitatively, what step 4.2's
flat transfer numbers already suggested: the invariance training did not measurably reduce this
model's reliance on color as a compatibility proxy.

Two other numbers in this table are worth flagging, not directly about this phase's hypothesis
but relevant to interpreting it: **raw SigLIP itself has the highest color-score correlation on
Amazon (0.391)** — higher than any trained model — meaning plain visual similarity already
substantially tracks color similarity in Amazon product photography (unsurprising: same-colored
products often are also similar in other ways). And phase 8's Amazon-trained model has *higher*
Polyvore-domain color reliance in absolute terms than phase 9/10's Polyvore-trained models show
on Amazon, though its Polyvore correlation is slightly negative — these cross-domain-application
numbers are secondary readings, included for completeness, not the phase's main comparison.

## Why the invariance loss had so little effect: a mechanistic check

To understand the null result rather than just report it, I checked whether frozen SigLIP's own
embedding space was already close to hue-invariant *before* any training. Sampling 5,000
train/val items: raw SigLIP's cosine similarity between an item's original and hue-perturbed-twin
embedding averages **0.936** (std 0.040), versus **0.532** (std 0.073) for random unrelated item
pairs. In other words, **SigLIP was already substantially invariant to a pure hue-channel shift
before any training in this project touched it** — shape, texture, and lighting dominate its
representation far more than raw hue does. The invariance loss's own training curve confirms
this from the other direction: even at the very first evaluation (early in epoch 0), the
invariance loss term was already small (~0.04-0.05, i.e. ~95-96% cosine similarity for the
*trained projection's* twin pairs) and only crept up slightly (not down) over subsequent epochs
as the compatibility loss kept improving — there was very little "invariance gap" for this loss
term to close in the first place.

This reframes the phase 9 finding: **the color-coordination heuristic that hurt Amazon transfer
is probably not well described as "the model treats raw hue as part of item identity."** SigLIP
itself already largely ignores raw hue for identity purposes. What phase 9's model more likely
learned is a *relational* pattern — real Polyvore outfits are frequently color-matched or
color-themed (the red-boot-to-red-coat, rhinestone-to-sequin examples from phase 9's qualitative
check), so the compatibility projection learned that broader color-family/theme agreement between
*two different items* predicts "goes together." A pure hue-channel perturbation invariance
constraint on a *single* item's own representation doesn't touch that relational pattern at all
— it can leave an item's own embedding referring to "pink" essentially unchanged (since SigLIP
already does that) while doing nothing to stop the compatibility head from having learned
"pink pairs well with pink" as a cross-item rule during training on real, frequently color-matched
outfits.

## Step 5: qualitative check — confirms the quantitative finding directly

`qualitative_examples/` has 6 Amazon-domain transfer grids (the pink laundry bag
`B01FWDLMYC` plus 5 more from the fixed query list used since phase 8/9, for direct visual
comparability).

**The laundry bag case is essentially unchanged.** Phase 9's naive model's top-5 was
dominated by pink/magenta items regardless of category (pink baby shoe, a sizing-chart image,
a zebra-pattern hat, a pink tank top, a pink sneaker). Phase 10's color-invariant model's top-5
is *also* dominated by pink/magenta/purple items (a purple sneaker, a pink tank top, pink
scrub top, a pink-themed ad image, a pink-striped compression sock) — the same failure mode,
essentially unchanged in character. The watch query (`B005NGRC0W`) lost phase 9's one clear
success case (a watch winder/box at #4) — phase 10's top-5 for this query is all watches
(same-category, not a genuine complement) plus a military rank-insignia image. The costume
query (`B00505DPQG`) shows both phase 9 and phase 10 continuing to retrieve predominantly
red-themed items (Flash costume, pirate costume, an Avengers watch, red-themed socks, Wonder
Woman) — again, little visible change in character between the two models. Across all 6
re-examined queries, no query showed a clear, unambiguous shift away from color-matching toward
more genuinely category/context-appropriate results.

## Interpretation

The hypothesis is **not supported**. A simple invariance-via-augmentation loss on individual
items' own hue-shifted twins:
1. Cost a small but real amount of in-domain Polyvore accuracy (as anticipated), and
2. Did **not** measurably reduce reliance on color matching (step 4.3's near-identical
   correlations) or improve Amazon transfer (step 4.2's flat-to-slightly-worse numbers), and
3. Produced qualitatively near-identical failure cases to phase 9's naive model (step 5).

The mechanistic check above explains why: the perturbation targets a gap (raw hue sensitivity
in a single item's own embedding) that mostly didn't exist to begin with, because frozen
SigLIP already encodes hue-shifted twins as highly similar. The actual mechanism behind phase
9's color-coordination shortcut is more likely a *relational* pattern learned from Polyvore's
real, frequently color-matched outfits — "these two items' colors agree" as a cross-item signal
— which a per-item invariance constraint doesn't touch.

## Recommendation

**Do not carry this specific approach (single-item hue-invariance augmentation) forward** —
it costs real in-domain accuracy for no measurable transfer benefit, confirmed by three
independent lines of evidence (transfer Hit Rate, color-correlation diagnostic, and qualitative
inspection) rather than any one of them alone. Per the brief's own scoping, the natural next
idea (an adversarial color-removal / explicit color-family classifier the projection is
trained to fool) was deliberately not attempted in this phase and remains open — but this
phase's mechanistic finding suggests any future attempt should target the *relational*
color-agreement signal between item pairs directly (e.g. an explicit penalty on the correlation
between predicted compatibility and pairwise color similarity, closer to step 4.3's own
diagnostic, rather than a per-item single-image invariance constraint) rather than assuming
augmentation-based invariance on individual items will generalize to a pairwise relational
heuristic. Phase 8's Amazon-trained model remains the best available compatibility signal for
Amazon's own domain; phase 9's naive Polyvore-trained model remains the strongest signal for
Polyvore's own domain — this phase does not change either recommendation.

# Phase 14b: Honest Interpretation

## What this phase set out to do

Per the professor's redirect
(`week4/phase14b_outfittransformer_category_negatives.md`), point 1 of
seven: fix OutfitTransformer's negative sampling to be category-restricted,
matching what the actual CIR evaluation task requires, and establish
whether the fixed reproduction is credible enough to serve as the project's
main baseline going forward.

## Step 1: the fix itself worked exactly as intended

`negative_sampling_fix_summary.md`: category-match rate among training
negatives went from phase 14's diagnosed 13.1% (unrestricted in-batch
negatives) to 100.0%, verified fresh over 192,000 sampled negatives, 0
short rows. This part of the brief is fully resolved and not in question --
every negative used in this phase's training is drawn from the target's
own category, by construction.

## Step 2 and 3: two training runs were needed, not one

The first full training run (`negative_mode="mined"`, using CSA-Net's own
same-category SigLIP-mined candidate list) converged smoothly by every
surface-level measure -- val_loss looked as clean as phase 14's own
(-3.5281 vs phase 14's -3.5050) -- but scored `Recall@10=0.0051`, about a
quarter of phase 14's own already-broken 0.0201. Checking the actual
triplet-margin component of the loss (not just the reported number, which
is dominated by the uniformity regularizer) showed why: it never resolved,
plateauing with the margin violated by ~0.30 for the entire 100-epoch run.
The mined negatives -- each anchor's closest same-category items by raw
SigLIP similarity -- were too hard for this small 4-layer transformer's
triplet-margin loss to separate from the true target at all.

Rather than report that as the final result, I ran a second, cheap
ablation: identical setup, but negatives drawn uniformly at random from
the same category instead of by SigLIP-nearest-neighbor mining
(`02b_train_full_random_negatives.py`). This isolates whether the category
restriction itself is sound (as CSA-Net's own precedent, and this
project's own CSA-Net reproduction, both suggest it should be) once it
isn't compounded with maximal negative difficulty. The full trace of both
runs, including the internal D_pos/D_neg diagnostic tables, is in
`training_log.md`.

**This second run is the real fix.** `Recall@10=0.0588`, `Recall@30=0.1286`,
`Recall@50=0.1809` -- roughly 2.93x, 2.19x, and 1.99x phase 14's original
broken numbers respectively, and clearly better than run 1's mined-negative
attempt (7.7x to 11.5x higher depending on K). One genuinely useful side
lesson from watching both runs live: the internal D_pos/D_neg training
diagnostic tracked almost identically between run 1 and run 2 the whole
way through (both plateaued with the margin violated by ~0.30), giving
essentially no warning that the real retrieval quality would differ by an
order of magnitude. This project already learned a version of this lesson
in phase 14 itself (a well-converged training diagnostic there also failed
to predict Recall@K); it is reconfirmed here from the opposite direction.

## Step 4: is this credible enough to be "the main baseline" going forward

Applying the bar stated in the brief directly: a similar range to
CSA-Net's own reproduction (roughly 80 to 90 percent of its own published
numbers, consistent across K), not just "better than the broken version."

Run 2 (the one worth judging against this bar) reaches **61.4% / 71.6% /
82.3%** of OutfitTransformer's own published Recall@10/30/50. This is a
real, substantial fix -- run 2 now beats raw SigLIP at every K (0.0588 vs
0.0553, 0.1286 vs 0.1067, 0.1809 vs 0.1437), which no OutfitTransformer
configuration in this project had done before, and it closes most of the
gap to CSA-Net's own frozen-SigLIP reproduction (81.1% / 92.3% / 98.1% of
that reproduction's own numbers). But it does **not** clear the brief's
own bar: the ratio to OutfitTransformer's published numbers climbs steadily
with K (61% to 72% to 82%) instead of sitting flat, unlike CSA-Net's
reproduction, which landed at a consistent ~88% across all three K values
in phase 13b. A climbing ratio with K, rather than a flat one, is the same
pattern phase 14's own broken version showed (21% to 33% to 41%) --
weaker here, but the same shape, which suggests some of the same
underlying mechanism (the model doing comparatively better at looser,
higher-K rankings than at the strict top-10) is still present, just less
severely.

**Plain answer: not yet credible enough to stand as the sole main baseline
by the bar set in the brief, but it is a real and substantial improvement,
and it is now this project's best-performing OutfitTransformer-mechanism
result by a wide margin.** Given the magnitude of the shortfall is real
but no longer severe (an 18-39 percentage-point gap to the "consistent
80-90%" bar, not the 4-5x gap phase 14's original broken version showed),
and given this result is on a frozen backbone -- exactly the setup CSA-Net's
own reproduction needed 20 epochs of ResNet18 fine-tuning to close a
similar gap in its own case (phase 13/13b) -- **a full, non-frozen-backbone
fine-tune is a reasonable and likely-necessary next step** if a tighter
match to OutfitTransformer's published numbers is required, rather than
further tuning of the frozen-backbone negative-sampling procedure alone.
This flag is raised plainly, per the brief's own instruction, rather than
treating this run's real improvement as sufficient on its own.

## What is settled and what is still open

**Settled**: the category-restriction diagnosis from phase 14 was correct
-- fixing it produces a large, real improvement (run 2). The specific
implementation matters enormously: mining maximally-hard same-category
negatives for this small transformer's triplet-margin loss actively hurts
(run 1), while simple random same-category sampling delivers the intended
fix. The project's internal D_pos/D_neg training diagnostic is not a
reliable stand-in for real Recall@K here, in either direction -- a
reconfirmed lesson, not a new one.

**Open**: whether a full fine-tune (unfreezing the SigLIP backbone, or at
minimum a larger/differently-regularized transformer) closes the remaining
18-39 percentage-point gap to OutfitTransformer's own published numbers.
Not attempted in this phase -- out of scope per the brief's own "do not do
yet" list until the fix itself was verified, which it now has been. This
is the natural next step if a tighter baseline match is needed before
moving to the professor's later points (3-7: feature weighting, SigLIP
variations, revisiting the dial only if it helps CIR, and the final
cross-method comparison).

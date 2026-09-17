# Phase 1b Notes: Category-Balanced Resample (quota=175, redo)

This replaces an earlier phase 1b run that used quota=73/category (~803
sampled, 751 final). That run is fully superseded and its folder was
deleted; results from it are referenced below only for comparison, since
they showed the same technique ranking and are useful context for how
sample size affects the numbers. Phase 1 (the original skewed snowball
sample, 776 products) is untouched and remains the baseline.

## Why redo with a larger quota

The quota=73 run had two problems: several category-level Hit Rate numbers
rested on small N (as low as 41 queries for Women, after image download
losses), and ground-truth overlap dropped to 6.0% (vs. phase 1's 17.0%) --
73 products/category wasn't deep enough for much of each product's real
`also_buy`/`also_viewed` network to survive into the sample. This run uses
**quota=175/category** to get deeper connectivity and larger per-category
N.

## Step 1: category inventory (reused, not recomputed)

Same 11 department-level categories as the first phase1b run (from
`categories[1]`, 383,604 total qualifying products -- image present +
relatedness present). Availability unchanged: smallest category
(Uniforms, Work & Safety) has 695 qualifying products, comfortably above
the new 175 quota. No category was expected to be scarcity-limited, and
none was.

## Step 2/3: stratified sampling, quota=175

All 11 categories hit their quota exactly, with no scarcity issues:

| Category | Sampled | Available |
|---|---|---|
| Women | 175 | 167,348 |
| Men | 175 | 105,731 |
| Novelty & More | 175 | 43,963 |
| Girls | 175 | 18,005 |
| Costumes & Accessories | 175 | 12,731 |
| Baby | 175 | 10,825 |
| Boys | 175 | 10,712 |
| Luggage & Travel Gear | 175 | 9,196 |
| Shoe, Jewelry & Watch Accessories | 175 | 3,401 |
| Traditional & Cultural Wear | 175 | 997 |
| Uniforms, Work & Safety | 175 | 695 |

Combined: 1,925 products before image download losses. 1,872 got a valid
downloaded image (53 dead links from the 2018 crawl, a 2.75% failure rate
-- notably lower than the quota=73 run's ~6.5%, just the luck of which
specific links landed in each sample this time).

Final category distribution (1,872 products) is the most balanced of any
run so far -- 7.6% to 9.3% per category, vs. phase 1's 50.9/42.5/7 split:

| Category | Count | % of sample |
|---|---|---|
| Girls | 175 | 9.3% |
| Luggage & Travel Gear | 175 | 9.3% |
| Shoe, Jewelry & Watch Accessories | 175 | 9.3% |
| Costumes & Accessories | 174 | 9.3% |
| Baby | 174 | 9.3% |
| Boys | 174 | 9.3% |
| Traditional & Cultural Wear | 174 | 9.3% |
| Men | 173 | 9.2% |
| Novelty & More | 171 | 9.1% |
| Uniforms, Work & Safety | 165 | 8.8% |
| Women | 142 | 7.6% |

Women again lost the most to download failures in absolute terms (33 of
175), but landed at N=142 this time vs. N=41 in the quota=73 run --
Women's numbers below are now on much firmer footing.

## Step 4: relatedness overlap check

**Overall: 8.4% (9,450 of 113,002 references), between phase 1's 17.0% and
the quota=73 run's 6.0%, as expected** -- more depth per category than
before, but still nowhere near phase 1's density since phase 1 grew deep
into just 2 dominant categories rather than spreading across 11. All 11
categories cleared the trustworthy threshold (>=20 references; actual
range 145-2,639 refs per category), no category flagged.

| Category | Overlap refs | Total refs | Overlap % |
|---|---|---|---|
| Men | 2,639 | 13,762 | 19.2% |
| Shoe, Jewelry & Watch Accessories | 1,214 | 10,095 | 12.0% |
| Novelty & More | 1,052 | 9,279 | 11.3% |
| Luggage & Travel Gear | 992 | 10,702 | 9.3% |
| Girls | 746 | 9,434 | 7.9% |
| Costumes & Accessories | 805 | 12,854 | 6.3% |
| Boys | 556 | 9,144 | 6.1% |
| Baby | 638 | 12,553 | 5.1% |
| Women | 435 | 11,526 | 3.8% |
| Traditional & Cultural Wear | 228 | 6,536 | 3.5% |
| Uniforms, Work & Safety | 145 | 7,117 | 2.0% |

Notably, Men's overlap density (19.2%) actually *exceeds* phase 1's
overall 17.0% -- Men's co-purchase graph is dense enough that even a
175-product slice captures a lot of real structure. Uniforms/Work/Safety
and Traditional & Cultural Wear sit under 4%, the sparsest of the 11 --
worth remembering these two categories' Hit Rate numbers are measured
against the thinnest ground truth, even though they clear the
"trustworthy" reference-count threshold.

## Results: overall

| Technique | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| resnet50 | 0.377 | 0.438 | 0.134 | 0.100 |
| clip_vit_b32 | 0.405 | 0.479 | 0.141 | 0.107 |
| fashionclip | 0.468 | 0.540 | 0.180 | 0.136 |
| siglip_base | 0.502 | 0.578 | 0.191 | 0.144 |

## Comparison across all three sample constructions

| Technique | Phase 1 (skewed, N=776) HR@5 | Phase 1b quota=73 (N=751) HR@5 | Phase 1b quota=175 (N=1872) HR@5 |
|---|---|---|---|
| resnet50 | 0.524 | 0.398 | 0.377 |
| clip_vit_b32 | 0.554 | 0.422 | 0.405 |
| fashionclip | 0.655 | 0.482 | 0.468 |
| siglip_base | 0.668 | 0.530 | 0.502 |

**The technique ranking holds exactly across all three runs: SigLIP >
FashionCLIP > CLIP ViT-B/32 > ResNet50, on every metric, every time.**
This is now confirmed under three different samples with very different
category composition and size -- about as robust a conclusion as this
phase can produce.

**Important interpretive point for the report:** absolute Hit Rate@5 fell
*again* going from quota=73 to quota=175 (e.g. SigLIP 0.530 -> 0.502),
even though overlap density *increased* (6.0% -> 8.4%) and per-category N
got much healthier. This looks counterintuitive but has a clean
explanation: Hit Rate@K asks whether the true related item lands in the
**top K out of the whole candidate pool**, and the candidate pool per
category more than doubled (73/category -> 175/category). More candidates
competing for the same top-5 slots makes it statistically harder to land
a hit at fixed K, independent of embedding quality -- this is a standard
property of Recall@K-style metrics, not evidence that a larger, more
representative sample makes embeddings "worse." **Absolute Hit Rate
numbers are only comparable across runs with the same candidate-pool size
per query; the technique ranking (which is what phase 1's core conclusion
rests on) is unaffected by this and has now been reproduced three times.**

## Per-category findings (see `report_summary_table.md` for the full table)

- **SigLIP wins outright in 10 of 11 categories on Hit Rate@5.** The one
  exception is **Boys**, where FashionCLIP narrowly leads (0.534 vs.
  0.529) -- close enough to be noise, and in fact reverses at K=10 (SigLIP
  0.569 vs. FashionCLIP 0.563), so it's best read as a near-tie rather
  than a real exception to the ranking, just the one category where
  SigLIP's lead isn't clean at every K.
- **Men and Novelty & More are consistently the easiest categories** for
  every technique (Men: 0.647-0.723 HR@5; Novelty & More: 0.503-0.684),
  well above their technique's overall average, and Men also has the
  highest overlap density (19.2%) of any category -- likely not a
  coincidence: denser real co-purchase structure gives retrieval more to
  find.
- **Uniforms, Work & Safety and Traditional & Cultural Wear are
  consistently the hardest** (0.139-0.276 HR@5 across techniques) --
  these are also the two lowest-overlap categories (2.0%, 3.5%), so this
  looks more like a sparse-ground-truth effect than an embedding failure
  specific to these product types. Worth re-checking with a deeper,
  category-specific sample if these categories matter for a later phase.
- Girls, which was the hardest category in the quota=73 run, moved to
  mid-table here (0.423-0.509 HR@5) once N grew from 73 to 175 -- a
  reminder that small-N category comparisons (like quota=73's) can be
  noisy enough to change which category looks "hardest."

## Report-ready artifact

`report_summary_table.md` consolidates category, N, overlap refs/%, and
Hit Rate@5 for all four techniques (plus the winning technique) into one
table -- meant to be used directly in the technical report rather than
cross-referencing `results_table.md` and `category_breakdown.md`
separately.

## Open questions for later phases

- Is Uniforms/Work/Safety and Traditional & Cultural Wear's weak
  performance really about sparse ground truth, or does it reflect a
  genuine visual-similarity-does-not-imply-co-purchase gap for these
  product types? Would need a deeper single-category sample to separate
  the two explanations.
- The Boys category is the one place the SigLIP > FashionCLIP ordering
  doesn't cleanly hold -- not concerning on its own, but worth keeping in
  mind if a later phase specifically targets children's categories.
- Given that Hit Rate@K magnitude is sensitive to candidate-pool size
  (this phase's main methodological finding), any future phase that wants
  to compare Hit Rate@K across different sample sizes should either hold
  the candidate pool size fixed or normalize for it explicitly.

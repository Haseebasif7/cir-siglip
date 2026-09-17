# Phase 3 Notes: Popularity Bias and Catalog Coverage EDA

## What this phase was for

Professor's feedback flagged two risks with a purely visual-similarity recommender:
overspecialization (near-duplicate results) and popularity bias (repeatedly surfacing
already-popular items over the long tail). This phase is diagnostic only -- measuring
whether/how strongly these show up, using data and embeddings already on disk. No new
model runs, no hybrid recommender design (that's later).

## Step 1: catalog-wide popularity lookup

Single streaming pass over the full `meta_Clothing_Shoes_and_Jewelry.json.gz` (same file
used since week 1), tallying how many times each asin appears inside any *other* product's
also_buy/also_viewed.

- **Ran fast, no memory issues**: 189.1s, peak RSS 465 MB. Much faster than phase 1b's
  equivalent streaming pass (260s) plus everything else that pass also did (sampling +
  1925 image downloads, 651.9s total) -- this pass does less per record (no image URL
  checks, no record storage beyond an int counter), which shows in the lower memory
  footprint despite covering more asins. Confirms the standing rule that indexing *small*
  per-key data (a counter) across the whole catalog is safe; it's storing full records
  per key that was the earlier memory risk (see phase 1b's own notes on that).
- Network throughput was uneven mid-run (dropped to ~800 lines/s around the 1:08 mark,
  recovered to 14-17k lines/s by the end) but never errored or needed a retry. Noting
  this since the standing rule flags this host as occasionally dropping connections --
  this run just had a slow patch, not a drop.
- **Surprise: 3,777,545 unique asins found, not ~2.68M.** Only 2,685,059 product *records*
  were processed, but every asin ever referenced inside anyone's also_buy/also_viewed also
  gets a lookup entry, even if that asin never appears as its own record in this
  category-specific file (e.g. it's a real product, just not one classified under
  Clothing/Shoes/Jewelry, or it's delisted). ~1.09M such "dangling reference" asins exist.
  They still get a legitimate ref_count and tier -- the edge is real even though we never
  saw that product's own metadata row -- but it means "the catalog" for popularity-lookup
  purposes is a bit broader than "the 2.68M records," worth remembering if a later phase
  is surprised a ground-truth or retrieved asin's tier came from an asin with no title/image
  in this file.
- Design decision: also_buy and also_viewed are unioned *within* a single referencing
  product before counting, so a neighbor listed in both doesn't get double-counted as two
  references from the same referrer. Treats "referenced by product X" as one boolean event
  per X, not per-field.

## Step 2: head/mid/tail tiers

Rank-based (true quantile) split over all 3,777,545 asins sorted descending by ref_count:
head = top 20% by rank, mid = next 30%, tail = bottom 50%.

| Tier | N | % of catalog | ref_count range | median |
|---|---|---|---|---|
| head | 755,509 | 20.0% | 1-4035 | 5.0 |
| mid | 1,133,263 | 30.0% | 0-1 | 0.0 |
| tail | 1,888,773 | 50.0% | 0-0 | 0.0 |

- **69.1% of the entire catalog (2,611,025 asins) has ref_count == 0** -- never appears in
  anyone's also_buy/also_viewed at all. This is the real shape of the long tail: it's not
  a thin tail, it's the majority of the catalog.
- As expected from that skew, the mid/tail boundary ties heavily (both ranges bottom out
  at 0), so the reported ranges overlap between mid and tail. Per the phase brief, this is
  reported as-is rather than forced into an artificially clean split -- the overlap itself
  is the finding (the distribution doesn't have enough distinct values at the tail to
  divide cleanly).
- Long-tail plot (`plots/long_tail_distribution.png`): classic Zipf shape on log-log axes
  (y offset by +1 to keep zero-count products visible on a log scale).

## Step 3: full retrievals

Reused phase 1b's `retrieval_results.json` directly rather than recomputing from
embeddings -- it already had the exact top-10 cosine-similarity lists for all four
techniques over the same 1,872-product sample. Reformatted (dropped phase 1b's own
hit-rate bookkeeping fields, kept asin/category/retrieved_top10) into
`data/full_retrievals.json`.

## Step 4-5: popularity bias analysis and visualizations

**The most important finding is about the *sample*, not the techniques**: the 1,872-product
phase 1b sample is itself 75.1% "head" tier by the full-catalog definition (see
`bucket_composition.md`), vastly more concentrated than the 20% head would be under
catalog-neutral sampling. This is very likely a direct consequence of BFS/snowball sampling
through the also_buy/also_viewed graph (used deliberately since phase 1, per the standing
rule against independent random sampling, to keep real ground-truth edges inside the
sample) -- products with many relatedness edges are, almost by definition, more likely to
get pulled into a BFS walk, and having many edges *is* what ref_count/popularity measures.
**This means every earlier phase's sample has likely been popularity-skewed all along; it
was never visible before because catalog-wide reference counts hadn't been computed until
this phase.** Flagging as a new open thread below.

Given that baseline, here's what retrieval looks like relative to it, for all four
techniques (`results_table.md`, `bucket_composition.md`, `plots/bucket_composition_*.png`):

- **Retrieved composition tracks the baseline closely** (76-78% head across all four
  techniques, vs 75.1% baseline) -- none of the four techniques is adding much *additional*
  popularity skew on top of what the sample already has. FashionCLIP and CLIP ViT-B/32 are
  marginally closest to baseline; SigLIP is the furthest above it (78.0% vs 75.1%), though
  the gap is small.
- **Ground truth is the most head-skewed of all: 96.2% head, 3.8% mid, 0.0% tail.** Real
  also_buy/also_viewed targets in this catalog concentrate on popular items far more than
  either the sample or any retrieval technique does. Read together with the point above,
  this actually cuts against the "the model is the source of popularity bias" framing --
  here, the ground truth itself is more popularity-concentrated than what any technique
  retrieves. A model trying to hit these targets more accurately would, if anything, need
  to skew *more* toward head items, not less. Worth surfacing to the professor directly:
  in this catalog, popularity bias may be a property of real co-purchase behavior, not
  (primarily) an artifact of visual-similarity retrieval.
- **ARP** is similar across techniques (FashionCLIP lowest at 38.5, SigLIP/ResNet50 highest
  around 42-43) -- a real but modest spread, consistent with the composition tables above.
- **Catalog coverage is high for every technique**: 89-94% @top-5, 94-98% @top-10. This
  needs a second look before taking at face value (see below).

### Second look: is high coverage actually informative here?

With 1,872 queries each contributing 5 top-5 slots, there are 9,360 recommendation slots
chasing only 1,872 possible items -- meaning even a fairly repetitive retrieval process
would likely touch most of the catalog at least once just from slot volume, so "coverage"
as a purely binary ever-recommended fraction is a weak diagnostic at this queries:items
ratio. Ran a supplementary (not part of the required deliverables, done as a sanity check)
concentration read on the same top-5 pools:

| Technique | Max single-item frequency | Top-10 items' share of all slots |
|---|---|---|
| ResNet50 | 26 / 9360 | 2.2% |
| CLIP ViT-B/32 | 28 / 9360 | 2.7% |
| FashionCLIP | 23 / 9360 | 2.3% |
| SigLIP | 40 / 9360 | 2.4% |

No technique has its most-repeated items dominating the slot pool (SigLIP's single most-
frequent item, 40 occurrences, is still under 0.5% of all slots). So the high coverage
numbers aren't masking a small dominant clique -- the repetition/overspecialization signal
genuinely looks mild in this sample, by both readings together, not just an artifact of
coverage's insensitivity at this scale. Still, recommend that any future phase drilling
further into overspecialization track a concentration/Gini-style measure directly rather
than relying on binary coverage alone, since the binary version can look reassuring for a
reason unrelated to actual diversity when n_queries*k approaches n_items.

## Open questions / anything needing a second look

1. **New**: the phase 1b BFS/snowball sample is ~75% head-tier by catalog-wide reference
   count -- far from catalog-neutral. Every retrieval conclusion in phases 1/1b/2/3 that
   used this sample was implicitly evaluated on a popularity-skewed subset of the catalog.
   Not necessarily wrong (BFS was the right call to get real ground-truth edges at all),
   but any future phase claiming a technique or fix "works on the catalog broadly" should
   check this. Reweighting or a catalog-proportional resample hasn't been done.
2. Ground truth being more head-skewed than any technique's retrieval (96.2% vs 75-78%)
   suggests popularity bias here may be substantially a property of real co-purchase
   behavior in this catalog, not an artifact introduced by visual similarity retrieval --
   worth confirming this reading holds if a future phase samples differently.
3. Coverage-as-defined is a weak diagnostic when n_queries*k >> n_items (see above) --
   flagging for any future phase that leans on it.
4. Dangling references (~1.09M asins referenced but with no own record in this file) are
   included in the catalog-wide tier lookup with real ref_counts. Fine for tiering, but
   worth remembering these asins have no title/image/category in `sample_data.csv`-style
   files if a later phase tries to look one up by asin and expects full metadata.

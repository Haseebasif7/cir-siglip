# Phase 6: Splitting also_buy by Visual Similarity

## Context

Phase 5 found `also_viewed` is completely empty in this project's metadata
source (confirmed against the raw file directly, not a sampling artifact),
so the originally planned substitute (also_viewed) vs complement (also_buy)
split can't be tested. This phase reframes the question around the one real
signal available: does the visual similarity *within* `also_buy` edges show
a pattern consistent with two underlying relations (near-identical/substitute
vs visually-different/complement), without needing a new ground-truth field?

Diagnostic only, per the brief: no training, no re-scoring retrieval with a
proposed threshold. Reused phase 1b's exact 1,872-product sample and its
existing SigLIP (and, for the optional extension, FashionCLIP) embeddings —
no new downloads, no new model inference.

## Step 1: edge similarity computed

9,450 within-sample `also_buy` edges (source, target both in the 1,872-product
sample) — matches phase 5's overlap count exactly, as expected since it's the
same edge set. Cosine similarity computed per edge from phase 1b's SigLIP
embeddings. `data/edge_similarities_siglip.json` has the full per-edge list.

## Step 2: distribution shape — mixed signal, not clean bimodality

See `similarity_distribution.md` and `data/similarity_distribution_siglip.png`.

| Statistic | SigLIP | FashionCLIP (optional extension) |
|---|---|---|
| Mean | 0.671 | 0.596 |
| Median | 0.682 | 0.607 |
| Std dev | 0.146 | 0.187 |
| KDE peaks detected | 2 (~0.59, ~0.75) | 2 (~0.54, ~0.70) |
| Sarle's bimodality coefficient | 0.436 | 0.451 |

Both encoders show the same qualitative shape: a broad, right-skewed
continuous mass roughly spanning 0.3-0.9, with a modest shoulder rather than
a clean valley separating two humps. The KDE peak-counter registers "2 modes"
for both, but Sarle's bimodality coefficient stays below the 0.555 rough
threshold for both — the two checks disagree, which itself is informative:
this is not a clean two-population split, it's a single continuous
distribution with a soft asymmetry. That the shape replicates almost
identically under a second, independently-trained encoder (FashionCLIP) rules
out "this is a SigLIP-specific quirk" — whatever structure exists here is a
property of the also_buy edges themselves, not an encoder artifact. It just
isn't strong enough structure to call bimodal.

## Step 3: qualitative check — extremes are dominated by artifacts, not a clean pattern

Full detail in `qualitative_examples/README.md`. Summary:

- **All 10 highest-similarity pairs (sim = 1.0000) turned out to be
  byte-for-byte identical image files** (verified by MD5 hash) — the same
  seller photo reused across size/color/pack-size variant ASINs. Checked
  further: of 9,450 edges, 61 (0.65%) score similarity > 0.999, and all 61
  are byte-identical images. This is a real pattern, but it's measuring photo
  reuse across catalog listings, not "genuinely similar but visually distinct
  near-duplicate products" — the qualitative signal the phase set out to
  check for at the high end doesn't hold up as intended.
- **6 of the 10 lowest-similarity pairs involve a corrupted product record**
  (title is a scraped JavaScript snippet, `var aPageStart = ...`, and the
  associated image is a sizing-chart graphic, not a product photo — affects
  16/1,872 products in the sample, 0.85%, concentrated in Costumes &
  Accessories). Their low similarity is a data-quality artifact, not evidence
  of a real complement relationship.
- Of the remaining, non-corrupted low-similarity pairs: **one is a genuine,
  clean complement example** (a pirate wig paired with a pirate blouse —
  visually unrelated, plausibly bought together to complete a costume).
  The others read more like different/alternative costume choices
  (substitute-like: same occasion, different specific item) than functional
  complements.

**Honest read**: there is a real complement example in the data, so the
underlying phenomenon the phase is looking for isn't imaginary. But out of
20 extreme pairs inspected, only 1 is a clean complement example, and both
extremes are substantially contaminated by data-quality issues (image reuse
at the top, corrupted records at the bottom) that have nothing to do with
substitute/complement structure. This is a weak, noisy qualitative signal,
not a confirmation.

## Step 4: no split point is proposed

Neither check supports a real, usable split:
- The distribution is not cleanly bimodal (Sarle's coefficient below
  threshold for both encoders; no valley of low density between two humps,
  just a soft shoulder in an otherwise continuous mass).
- The qualitative extremes, where a split threshold would need to draw its
  clearest evidence, are dominated by artifacts unrelated to the
  substitute/complement distinction (duplicate photos, corrupted records),
  with only one genuinely clean complement example found across 20 inspected
  pairs.

**Saying this directly, per the brief: visual similarity alone does not
separate also_buy edges into substitute and complement groups in a way this
data supports.** Proposing a numeric threshold (e.g. "below 0.5 = complement")
would be fitting a story to noise — the shoulder around 0.55-0.65 is real but
too weak and too contaminated by artifacts to operationalize with any
confidence.

## Step 5: alternate also_viewed source — none confirmed, quick check only

See `also_viewed_alternate_source_check.md`. No alternate hosted copy with a
confirmed-populated `also_viewed` field was found in a bounded search. Useful
side-finding: the *2014* Amazon dataset (different, nested schema) did have
`also_viewed` populated, so the emptiness is specific to this project's 2018
metadata source, not "Amazon relatedness data in general." Kaggle mirrors
exist but are almost certainly re-hosts of the same source data and weren't
independently verified (out of scope for a quick check).

## Optional extension: FashionCLIP

Ran the distribution check (step 2 equivalent) for FashionCLIP as well, since
it was cheap (reused existing embeddings). Result: same shape, same
conclusion (see table above) — the pattern is not SigLIP-specific. Did not
repeat the full qualitative image-grid check (step 3) for FashionCLIP, since
the distribution-level replication already answers the generalization
question this extension was meant to check, and the primary finding (no
usable split) doesn't change by adding a second qualitative pass.

## Recommendation

**Do not operationalize a visual-similarity-based substitute/complement split
of also_buy edges.** The signal is too weak and too contaminated by data
artifacts to support a threshold, and this holds under two different
encoders. Combined with phase 5's finding (also_viewed is empty), this
project currently has no clean, data-supported way to distinguish substitute
from complement relationships using what's in the Amazon 2018 metadata as
currently sourced.

If the hybrid recommender still needs a substitute/complement axis, the
options raised in phase 5 remain the live candidates (a different populated
data source, a category-agreement heuristic within also_buy, or reframing
the goal around also_buy as a single relatedness signal) — none attempted
yet. Two additional, narrower observations from this phase worth carrying
forward regardless of that decision:
- **Duplicate/near-duplicate product images are a real, measurable
  phenomenon in this catalog** (0.65% of also_buy edges are exact photo
  reuse) — worth a dedicated dedup pass before any future phase that assumes
  each also_buy edge connects two independently-photographed products.
- **A small (0.85%) but real fraction of product records have corrupted
  titles/images from a scraping artifact** (JS snippet as title, sizing-chart
  image instead of a product photo) — worth a basic data-quality filter
  (e.g. flag titles matching common JS/HTML patterns) before any phase that
  depends on title text or assumes every image is a genuine product photo.

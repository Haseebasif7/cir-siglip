# Phase 2 Notes: Product-Region Cropping vs Template Latching

## Motivation (recap)

Phase 1's qualitative analysis found that all four frozen embedding
techniques sometimes retrieve based on shared product-photography
template/framing (e.g. "wrist + gray mannequin silhouette" studio shots)
rather than the product itself. The flagship example was query
`B019SPRFPA` (a Marc by Marc Jacobs crystal charm bracelet): every
technique's top matches were other items sharing the exact same wrist/gray
silhouette template, with zero real `also_buy`/`also_viewed` hits. This
phase tests VLCLIP's approach directly: crop to the product region before
embedding, see if it fixes this failure mode and improves retrieval more
generally, for the two best techniques from phase 1b (SigLIP, FashionCLIP).

**Headline result: the hypothesis is not supported.** Cropping does not
reliably fix template latching and has a net-negative to flat effect on
retrieval quality overall, hurts FashionCLIP consistently and non-trivially,
and does not improve -- in fact slightly worsens -- results in the exact
category (Shoe, Jewelry & Watch Accessories) that motivated the experiment.
The mechanism turned out to be more interesting than a simple "didn't work":
see below.

## Setup

- Reused the exact phase 1b sample (1,872 products, category-balanced
  across 11 departments) and its existing SigLIP/FashionCLIP embeddings --
  no resampling, so results are directly comparable to phase 1b's numbers.
- **Method A (rembg):** U2-Net background removal, crop to the foreground
  alpha mask's bounding box (padded 8%). Generic, no prompt.
- **Method B (Grounding DINO):** `IDEA-Research/grounding-dino-tiny` via
  `transformers` (not the original repo -- this transformers version ships
  a pure-PyTorch port with no custom CUDA ops, which is what made it
  practical to run locally on MPS at all). Prompt: `"product. clothing
  item."`, generic and not tuned per category. Crop to the highest-scoring
  detection's box, padded 8%.
- Both methods excluded (not fell back to the original for) any image
  where cropping clearly failed -- see "Crop failure rates" below.

### Two MPS-specific issues hit and fixed

- `torch.cummax`, used internally by Grounding DINO's text-token masking,
  isn't implemented on the MPS backend (`NotImplementedError:
  aten::_cummax_helper`). Fixed with `PYTORCH_ENABLE_MPS_FALLBACK=1`, which
  lets just that one op fall back to CPU while everything else stays on
  MPS.
- Batching multiple images through Grounding DINO in one forward pass
  caused MPS to OOM (batch=8 tried to allocate 17.7GB) -- deformable
  attention's memory scales badly with batch size on this backend. Ran
  one image at a time instead; ~1,872 images took ~2.5 hours this way.
  Fine for a one-time preprocessing pass, but worth knowing before trying
  to batch this model on Apple Silicon again.
- The crop job was accidentally interrupted partway through by an
  unrelated folder deletion/restore incident and had to be rerun from
  scratch (no resume logic existed) -- lost time, not lost correctness,
  since the rerun started clean.

## Crop failure rates

| Method | Attempted | Succeeded | Failed | Failure rate |
|---|---|---|---|---|
| A (rembg) | 1,872 | 1,869 | 3 | 0.2% |
| B (Grounding DINO) | 1,872 | 1,857 | 15 | 0.8% |

Both methods succeeded on the overwhelming majority of images. Method A's
3 failures were all degenerate/empty foreground masks. Method B's 15
failures were mostly `near_empty_crop` (14) plus 1 `no_detection`. Full
logs: `logs/crop_a_log.md`, `logs/crop_b_log.md`.

**Important limitation of the failure filters, found via the case study
below: neither filter catches a large-but-mislocalized box.** Both only
check crop area/dimensions against minimum thresholds, so a box that
confidently but wrongly covers most of the frame (e.g. background instead
of product) passes as a "success" with no way to distinguish it from a
correct wide product shot. This means the documented failure rates above
are a floor, not a ceiling, on how often Method B in particular put the
box in the wrong place.

## Common evaluable set

All comparisons (baseline, Method A, Method B, both encoders) are computed
on the intersection of asins that got a valid crop under *both* methods --
**1,854 of 1,872 products (99.0%)** -- so Hit Rate@K/Precision@K are never
confounded by a different candidate-pool size between variants (see
PROJECT_MEMORY.md's standing rule on this). Details:
`data/common_evaluable_set.md`.

## Results: overall

| Encoder | Variant | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|---|
| SigLIP | Uncropped (baseline) | 0.501 | 0.576 | 0.191 | 0.144 |
| SigLIP | Method A (rembg) | **0.508** | 0.574 | 0.199 | 0.150 |
| SigLIP | Method B (Grounding DINO) | 0.495 | 0.563 | 0.187 | 0.140 |
| FashionCLIP | Uncropped (baseline) | **0.464** | **0.538** | **0.179** | **0.136** |
| FashionCLIP | Method A (rembg) | 0.435 | 0.514 | 0.151 | 0.114 |
| FashionCLIP | Method B (Grounding DINO) | 0.440 | 0.498 | 0.164 | 0.122 |

- **SigLIP**: essentially a wash. Method A nudges Hit Rate@5 up very
  slightly (+0.007); Method B nudges it down slightly (-0.006). Neither
  effect is large enough to call cropping a real improvement.
- **FashionCLIP**: cropping clearly hurts, for both methods, on every
  metric. Method A costs -0.029 Hit Rate@5 (a ~6% relative drop); Method B
  costs -0.024. This is the opposite of what the hypothesis predicted.

### Per-query flip counts (why the aggregate numbers move the way they do)

| Encoder + method | Queries flipped 0→hit (improved) | Queries flipped hit→0 (worsened) | Net |
|---|---|---|---|
| SigLIP + Method A | 70 | 56 | +14 |
| SigLIP + Method B | 66 | 77 | -11 |
| FashionCLIP + Method A | 59 | 113 | **-54** |
| FashionCLIP + Method B | 66 | 111 | **-45** |

FashionCLIP loses roughly twice as many previously-correct queries as it
fixes, under both crop methods. SigLIP is much closer to a wash, with
Method A slightly ahead. This is a real, encoder-dependent effect, not
noise -- consistent across both crop methods for both encoders.

## Category breakdown: the motivating category doesn't improve either

Full table: `category_breakdown.md`. The category most relevant to the
original bracelet example, **Shoe, Jewelry & Watch Accessories**, actually
gets *worse* with cropping for both encoders and both methods:

| Encoder | Uncropped | Method A | Method B |
|---|---|---|---|
| SigLIP | 0.594 | 0.589 | 0.554 |
| FashionCLIP | 0.537 | 0.514 | 0.451 |

This directly undercuts the hypothesis at its most favorable test case --
if cropping were fixing template latching in jewelry/watch photography,
this category should have been the clearest beneficiary, and instead it's
one of the categories where cropping costs the most (FashionCLIP + Method
B: -0.086 Hit Rate@5, one of the largest drops of any category in the
table). A few categories do improve with cropping (e.g. Men with SigLIP +
Method A: 0.723 → 0.769), but there's no consistent per-category story --
it looks more like noise/redistribution than a systematic category effect.

## Why cropping mostly doesn't help: two mechanisms found

### 1. Most catalog images were already tightly framed -- little room for cropping to help

Measured crop-area-retained (crop pixel area ÷ original image pixel area)
on a random sample of 300 crops per method:

| Method | Mean area retained | Median | % retaining >90% (~no-op) | % retaining <30% (tight crop) |
|---|---|---|---|---|
| A (rembg) | 80.5% | 89.9% | 50% | 2% |
| B (Grounding DINO) | 69.8% | 79.0% | 44% | 17% |

Half of Method A's crops keep over 90% of the original image -- i.e. for
half the catalog, there simply wasn't much background to remove in the
first place (typical Amazon product photography is already a tight
product-on-plain-background shot, not a wide scene). Phase 1's bracelet
example, with its unusually large empty gray-silhouette region, was more
the exception in this sample than the norm. This alone limits how much a
generic crop can improve things on average, regardless of method quality.

### 2. When cropping does change the image a lot, it can destroy useful context, not just remove noise

The qualitative examples below show this directly: several of
FashionCLIP's worst regressions are cases where the uncropped baseline was
already retrieving well (4/5 or 5/5 real hits) using the *whole* photo --
watch face + band + wrist together -- and an aggressive crop that isolates
just the watch face converts a near-perfect result into a total miss. The
crop removes exactly the context (band material, case style, overall
silhouette) that the real co-purchase signal seems to have been keying on.
Cropping is not a pure noise-removal operation on this catalog; it also
removes real information some of the time.

## Case study: query B019SPRFPA (the original bracelet example)

Full results: `qualitative_examples/case_study_B019SPRFPA/` (`results.json`,
`grid_siglip_base.png`, `grid_fashionclip.png`). B019SPRFPA isn't part of
the phase 1b sample (different draw from the catalog than phase 1), so it
was processed as a one-off extra query: cropped with both methods,
embedded with both encoders, and matched against the phase 2 pool. Four
rows are compared per encoder: phase 1's original finding (776-product
pool), an uncropped-in-phase-2-pool reference, Method A, and Method B.

**Notable finding: Grounding DINO's crop for this exact query is dominated
by the gray mannequin silhouette itself, not the bracelet.** The crop
keeps ~79% of the original frame and is centered on the hip/thigh
silhouette region, with the bracelet visible only as a sliver at the edge
(see `query_cropB.jpg`). The prompt `"product. clothing item."`
apparently matched the stylized gray body shape as "a clothing item" --
the exact same template-latching visual pattern this phase set out to
remove ended up fooling the fix itself. This slipped past both automated
failure filters (area and dimension based) since the box isn't small or
degenerate, just wrong -- see the failure-filter limitation noted above.

Other observations from this case:
- **SigLIP's uncropped retrieval in the phase 2 pool correctly surfaces an
  actual bracelet** (`B015FH7R4O`, a sterling silver beaded bracelet) in
  its top 5 -- a real semantic match, even without cropping. Both crop
  methods lose this and replace it with earrings-only results (Method A)
  or, worse, military medals and rank insignia (Method B) -- small shiny
  metallic objects on a plain background, a new and different
  template-latch failure that cropping introduced rather than fixed.
- **FashionCLIP's crops shift the retrieval toward novelty character
  watches** (Disney, Marvel Captain America/Spider-Man kids' watches) --
  still "wrist accessory," but a worse semantic match to a designer crystal
  bracelet than the uncropped result.
- None of the four variants produce a ground-truth hit for this query (0/5
  everywhere) -- expected, since this asin's real `also_buy`/`also_viewed`
  neighbors mostly live in phase 1's sample, not phase 1b's, so this case
  study is read qualitatively, not as a hit-rate result.

**Conclusion for this specific case: cropping did not fix the
template-latching failure it was designed to address, and for Method B it
reproduced essentially the same class of failure (latching onto the
photographic template/background) using a different template.**

## General qualitative examples

Five more examples, automatically selected from FashionCLIP + Method A
(the encoder/method pairing with the largest overall Hit Rate@5 swing vs
baseline, so the most informative to look at): 2 improved, 2 worsened, 1
unchanged. Full images and picks: `qualitative_examples/general_examples/`.

- **Improved -- `B00GMA414S`** (LEGO Batman minifig watch): baseline 0/5
  hits (retrieves other LEGO watches, but none are real co-purchase
  matches); Method A gets 2/5 real hits (other LEGO franchise watches --
  Star Wars, Ninjago). The crop barely changes this image (already
  tightly framed), so the improvement here looks like minor
  re-ranking/normalization rather than a dramatic re-localization effect.
- **Improved -- `B00BOVD4ZM`** (blue-dial chronograph watch): baseline 0/5,
  Method A gets 2/5. Similar pattern -- small framing change, modest gain.
- **Worsened -- `B00UNCKVIG`** (U.S. Polo Assn. leather-band watch):
  baseline is a **perfect 5/5** -- all five retrieved items are real
  co-purchase matches, using the full wrist+band+case photo. Method A
  crops in tight on the watch face, and retrieval collapses to **0/5**,
  now matching on close-up dial/case macro shots from completely different
  watch lines. This is the clearest single illustration of mechanism #2
  above: the crop discarded the band/case context that the real signal
  depended on.
- **Worsened -- `B00O9ZM6WW`** (silver dress watch): baseline 4/5 real
  hits; Method A crop drops to 0/5, shifting toward black sporty/digital
  watches instead of the silver dress-watch line. Same mechanism as above.
- **Unchanged -- `B0010TP398`** (Kiwi shoe polish tin, top-down flat-lay
  shot): baseline 5/5, Method A crop also 5/5 (same five items, order
  shuffled slightly). The product already fills nearly the whole frame, so
  the crop is close to a pixel-level no-op here -- ties directly back to
  the area-retained finding above.

## Conclusion

**The template-latching hypothesis from phase 1 is not supported by this
experiment**, at least not for a generic (non-fine-tuned) cropping
preprocessing step:

1. Cropping does not fix the specific failure case that motivated it
   (B019SPRFPA) -- if anything, Grounding DINO's crop for that exact image
   reproduces the same template-latching failure mode by locking onto the
   gray silhouette background instead of the product.
2. Cropping's aggregate effect is flat for SigLIP and clearly negative for
   FashionCLIP, and it does not improve the jewelry/watch category the
   original example came from -- it makes that category worse for every
   encoder/method combination tested.
3. The mechanism is now reasonably well understood: (a) most catalog
   images were already tightly framed, so there wasn't much background
   noise to remove for the average product, and (b) when a crop does
   change the image substantially, it just as often destroys real
   context (how an item is worn/displayed together with its
   accessories) as it removes irrelevant background -- and the FashionCLIP
   results suggest this destructive effect outweighs the benefit more
   often than not.

This is a legitimate negative result, not a failed experiment: it rules
out a plausible, literature-backed fix (VLCLIP's approach) as a
general-purpose intervention on this catalog, and identifies *why* --
context removal is a double-edged operation, not a pure denoising step,
for frozen embeddings on typical Amazon product photography. That's useful
evidence against spending further effort on preprocessing-only fixes and
toward the deferred fine-tuning/VBPR-style options for the next phase,
where the model could in principle learn *when* to ignore background
rather than having it removed unconditionally.

## Open questions for later phases

- Would a *partial* crop (looser padding, or a crop that keeps some margin
  around the detected box rather than tight-cropping to it) recover the
  context that mechanism #2 shows gets lost, while still removing the most
  extreme background cases like the original bracelet example? Not tested
  here -- both methods used a fixed 8% padding.
- The Grounding DINO failure-filter blind spot (large-but-wrong boxes
  passing as "success") means Method B's true mislocalization rate is
  unknown, not just its documented 0.8% failure rate. A held-out visual
  audit of a random sample of Method B crops would be needed to estimate
  this properly before trusting Method B's numbers as a clean read on
  "grounding-based cropping" in general, vs. this pass's specific prompt/gaps.
- Does a fine-tuned/trained approach (deferred from this phase, e.g.
  triplet loss or VBPR-style) do better than either uncropped or cropped
  frozen embeddings? This phase's finding that context matters as much as
  noise-removal suggests a learned approach that can weigh visual features
  by their actual correlation with co-purchase, rather than any fixed
  preprocessing rule, is the more promising direction.

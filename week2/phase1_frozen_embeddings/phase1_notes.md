# Phase 1 Notes: Frozen Embedding Benchmark

## Sample construction (and a methodology bug caught mid-phase)

The first version of the sampling script drew 500 products as **independent
random picks** from the qualifying pool (image present + `also_buy`/`also_viewed`
present). That produced a sample with **zero** relatedness overlap: of 13,209
`also_buy`/`also_viewed` references across the sample, none pointed at another
product inside the same sample. This is expected in hindsight -- the
probability that two specific related products both land in a random
455-item subsample of a 2.68M-product catalog is negligible. Every
technique scored an identical Hit Rate@K of 0.000, which is not a real
result, just confirmation that the ground truth couldn't possibly be found.

Fix: rebuilt the sample with BFS/snowball sampling instead. Seed products
were chosen (fixed seed=42) from the qualifying pool, then also_buy/also_viewed
neighbors were pulled in and expanded from, growing a connected subgraph up
to 800 products. Because the full seed-eligible pool (383,604 products) was
already held in memory with its edges, the BFS itself ran in memory almost
instantly -- reaching the 800-product target using only in-pool neighbors,
with no separate network lookup needed for out-of-pool leaf products.

Result: 776 of the 800 sampled products got a valid downloaded image (24
dead links from the 2018 image crawl, logged in `logs/failed_downloads.log`).
**8,978 of 52,910 relatedness references (17.0%) now point inside the
sample** -- confirmed via the script's built-in overlap check before moving
on, so Hit Rate@K measures something real this time.

Two more engineering issues along the way, noted for anyone rerunning this:
- A naive first attempt at neighbor lookup indexed *every* product with a
  valid image (~2M+) in one dict for O(1) BFS lookups; RSS hit ~530MB at
  only 3.4% of the file, which extrapolates to ~15GB on this 16GB machine.
  Switched to indexing only the ~380k image+relatedness products instead.
- The metadata stream (1.5GB gzip, single HTTP connection) dropped
  mid-download twice (`ReadTimeoutError`, then `ProtocolError:
  IncompleteRead`) during development. Added a retry wrapper; the exceptions
  needed catching at the `urllib3` level too since reading directly from
  `resp.raw` bypasses `requests`'s own exception wrapping.

### Category representativeness caveat

Snowball sampling over-represents whichever categories are best-connected
in the co-purchase graph. Confirmed here: the final 776-product sample is
**50.9% Women's and 42.5% Men's** clothing/jewelry/accessories, with Girls,
Boys, Baby, Novelty, and Luggage together making up under 7% (full
breakdown in `data/category_distribution.md`). **The numbers below describe
retrieval quality for adult apparel/jewelry/accessories, not the full
breadth of the Clothing/Shoes/Jewelry catalog.** Worth correcting for
(e.g. category-stratified seeding) if a later phase needs a more
representative slice.

## Results

| Technique | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| resnet50 | 0.524 | 0.624 | 0.204 | 0.156 |
| clip_vit_b32 | 0.554 | 0.644 | 0.207 | 0.161 |
| fashionclip | 0.655 | 0.738 | 0.271 | 0.209 |
| siglip_base | 0.668 | 0.753 | 0.289 | 0.222 |

Clear, consistent ordering across all four metrics: **SigLIP > FashionCLIP >
CLIP ViT-B/32 > ResNet50**. The two modern contrastive vision-language
models outperform plain CLIP, which in turn beats ImageNet-supervised
ResNet50 -- matching the expectation from the literature review that
contrastive/fashion-aware pretraining transfers better to this kind of
visual co-purchase retrieval than classification pretraining does.

## Qualitative patterns (from `qualitative_examples/`)

**1. All four techniques frequently latch onto the product-photography
template/pose rather than the product itself.** Query `B019SPRFPA` (a
beaded charm bracelet, shot as a wrist held against a plain gray
mannequin-silhouette background) retrieves near-identical "wrist + gray
silhouette" shots for every technique -- other charm bracelets, but also,
for ResNet50, a completely unrelated shirt pair and a pair of earrings on
a card, just because the framing/pose loosely resembles the query in
ResNet50's more shape/texture-driven feature space. None of these were
ground-truth hits: the studio photography convention is shared across many
unrelated charm bracelet listings, but that doesn't mean customers actually
buy across them.

**2. Template similarity helps exactly when it happens to correlate with
real co-purchase behavior, and hurts when it doesn't.** Contrast the above
with query `B000GB1R96` / `B000GAYQKO` (steel watches on the same
gray-arm template): FashionCLIP and SigLIP both return 5/5 real hits here,
because the watches that share this exact photography setup are, in this
case, genuinely from the same product family/seller line that customers
co-purchase. The failure mode isn't "using the template as a signal" per
se -- it's that template similarity is a spurious correlation with ground
truth for some product types (bracelets, evidently) and a real one for
others (this watch line), and none of these frozen embeddings can tell the
difference.

**3. ResNet50 shows the weakest fine-grained discrimination.** Query
`B0161ISG9G` is a delicate silver lotus-flower pendant necklace; ResNet50's
top misses include a bronze tribal/gothic face-pendant on a black cord, a
single blue stud earring against an ear diagram, dangling crystal
earrings, and a rose-gold watch. The common thread is coarse shape (thin
curved line + small pendant-shaped object at one end) while color,
material, and style are ignored entirely -- consistent with ImageNet
classification pretraining building shape/texture features that aren't
tuned for fine jewelry style discrimination, unlike the fashion/contrastive
models.

**4. CLIP ViT-B/32 does noticeably well on watches specifically.** Query
`B000EI858M` (a macro shot of a gold watch caseback/movement) returns 5/5
real hits, all other watch casebacks/watch shots. Where the product has a
strong, consistent, textured visual signature (metal watch mechanisms),
even plain CLIP performs on par with the fashion-tuned models; the gap
between techniques narrows for product types with less ambiguous visual
identity.

## Open questions for later phases

- How much of FashionCLIP/SigLIP's edge is genuine fine-grained fashion
  understanding vs. simply having seen more product-photography-style
  images in pretraining (i.e., better prior on this specific photography
  convention, not necessarily fashion semantics)?
- The template-latching failure mode (pattern 1/2 above) suggests grounding
  /cropping to the product region (ignoring background/mannequin) could be
  a meaningful improvement direction -- this matches what the Mercari and
  VL-CLIP papers from the lit review tried.
- Category skew (point above) means these numbers should be re-checked
  against a more balanced sample before drawing conclusions that generalize
  beyond women's/men's apparel and jewelry.

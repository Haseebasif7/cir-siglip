# Phase 4 vs. Amazon: Does the Technique Ranking Hold?

## The core question

Phase 1b (Amazon, `also_buy`/`also_viewed` ground truth, category-balanced 1,872-product
sample) found: **SigLIP (0.502 HR@5) > FashionCLIP (0.468) > CLIP ViT-B/32 (0.405) >
ResNet50 (0.377)**.

Phase 4 (DeepFashion In-shop, official query/gallery protocol, 14,218 queries against
12,612 gallery images) found: **SigLIP (0.894 Recall@5) > FashionCLIP (0.851) > CLIP
ViT-B/32 (0.673) > ResNet50 (0.459)**.

**The ranking holds exactly, in the same order, on both datasets.** This is now the
most-validated conclusion in the project: it has held across 3 Amazon sample
constructions (phase 1's original 776-product sample, phase 1b's quota=73 and quota=175
resamples) and now on a completely different dataset with a completely different kind of
ground truth (clean same-item photography matches, not behavioral co-purchase signal).

## Side-by-side numbers

| Technique | Amazon HR@5 (phase 1b) | DeepFashion Recall@5 | Absolute gap | Relative gap |
|---|---|---|---|---|
| SigLIP | 0.502 | 0.894 | +0.392 | +78% |
| FashionCLIP | 0.468 | 0.851 | +0.383 | +82% |
| CLIP ViT-B/32 | 0.405 | 0.673 | +0.268 | +66% |
| ResNet50 | 0.377 | 0.459 | +0.082 | +22% |

(See `phase4_notes.md` for why these two columns are not strictly the same
*measurement* despite both being "top-5 hit" metrics -- the gap sizes below are still a
valid comparison of magnitude, just not a claim that the two numbers were produced by an
identical procedure.)

## What's notable beyond "the ranking holds"

1. **Every technique scores higher on DeepFashion than on Amazon**, which is expected --
   DeepFashion's ground truth (same physical item, different photo) is a much easier,
   cleaner signal than Amazon's `also_buy`/`also_viewed` (different items a real customer
   happened to also buy or view, filtered through popularity effects phase 3 documented).
2. **The gap is not uniform across techniques -- it's much smaller for ResNet50 (+22%)
   than for the other three (+66% to +82%).** ResNet50's Amazon and DeepFashion scores
   are relatively close to each other (0.377 vs 0.459), while SigLIP and FashionCLIP jump
   dramatically under DeepFashion's clean ground truth. Read together, this suggests
   Amazon's noisy/popularity-skewed behavioral ground truth (phase 3's finding) may be
   *compressing* how much better the fashion-aware, contrastively-pretrained encoders
   (SigLIP, FashionCLIP) actually are at pure visual similarity -- their real advantage
   over generic ImageNet features looks much larger once evaluated on a task that only
   requires pure visual matching. ResNet50's generic ImageNet features stay weak
   regardless of which ground truth is used, so its score barely moves.
3. **The best-vs-worst spread widens a lot on the cleaner benchmark**: Amazon's
   best-minus-worst gap is 0.125 HR@5 (SigLIP - ResNet50); DeepFashion's is 0.435 Recall@5.
   If the goal is to argue "encoder choice matters," DeepFashion makes a far more dramatic
   case for it than Amazon's behavioral signal does on its own.
4. **DeepFashion's absolute numbers are plausible against the published literature**
   (frozen/zero-shot contrastive encoders scoring 0.66-0.74 Recall@1 and generic
   ImageNet-supervised features scoring meaningfully lower is consistent with what's
   reported elsewhere for non-fine-tuned baselines on this benchmark) -- a useful sanity
   check that the extraction/eval pipeline itself is sound, independent of the Amazon
   comparison.

## Bottom line

The technique ranking is not an Amazon-specific artifact -- it reproduces on a dataset
with a fundamentally different kind of ground truth (clean identity-match vs. noisy
behavioral signal), which is exactly the test the professor asked for. If anything,
DeepFashion suggests the encoder-choice effect is *larger* than Amazon's numbers alone
would indicate, since Amazon's ground truth noise (and phase 3's popularity skew) likely
suppresses the visible gap between techniques.

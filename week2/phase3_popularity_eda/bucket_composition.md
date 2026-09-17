# Phase 3: Head/Mid/Tail Composition -- Sample vs Retrieved vs Ground Truth

Tiers are catalog-wide (from step 1+2's full metadata pass), not sample-relative.

## Baseline: composition of the 1,872-product sample itself

This is the reference point -- if retrieval were popularity-agnostic, retrieved items should roughly match this composition.

| Tier | % of sample |
|---|---|
| head | 75.1% |
| mid | 8.8% |
| tail | 16.2% |

(n=1872 sample products with a known tier)

## Ground truth: composition of real also_buy/also_viewed targets

Aggregated across every query's actual relatedness edges (not the retrieved items) -- shows what the "correct answers" look like in terms of real popularity, independent of any model.

| Tier | % of ground truth targets |
|---|---|
| head | 96.2% |
| mid | 3.8% |
| tail | 0.0% |

(n=113002 ground-truth also_buy/also_viewed references with a known tier)

## ResNet50: retrieved composition (top-5, top-10)

| Tier | % of top-5 retrieved | % of top-10 retrieved |
|---|---|---|
| head | 76.2% | 75.7% |
| mid | 8.6% | 8.7% |
| tail | 15.2% | 15.6% |

## CLIP ViT-B/32: retrieved composition (top-5, top-10)

| Tier | % of top-5 retrieved | % of top-10 retrieved |
|---|---|---|
| head | 75.1% | 74.6% |
| mid | 10.2% | 10.6% |
| tail | 14.7% | 14.9% |

## FashionCLIP: retrieved composition (top-5, top-10)

| Tier | % of top-5 retrieved | % of top-10 retrieved |
|---|---|---|
| head | 76.0% | 76.0% |
| mid | 9.4% | 9.5% |
| tail | 14.6% | 14.5% |

## SigLIP: retrieved composition (top-5, top-10)

| Tier | % of top-5 retrieved | % of top-10 retrieved |
|---|---|---|
| head | 78.0% | 77.8% |
| mid | 9.0% | 8.8% |
| tail | 13.0% | 13.3% |

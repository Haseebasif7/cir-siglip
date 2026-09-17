# Phase 16d: Adjacent vs Distant Alpha Overlap

Same method as phases 16/16c (phase 12d's method), full pairwise top-10 overlap matrix across all 11 alpha values, all 1872 phase 1b queries.

## Mean overlap as a function of |delta alpha|

| |delta alpha| | Mean top-10 overlap |
|---|---|
| 0.1 | 0.8595 |
| 0.2 | 0.7271 |
| 0.3 | 0.6018 |
| 0.4 | 0.4914 |
| 0.5 | 0.4027 |
| 0.6 | 0.3377 |
| 0.7 | 0.2931 |
| 0.8 | 0.2663 |
| 0.9 | 0.2530 |
| 1.0 | 0.2486 |

**Adjacent steps (delta=0.1): mean overlap = 0.8595**
**Most distant (delta=1.0): mean overlap = 0.2486**
**Gap (adjacent - distant): 0.6110**

## Three-way comparison

- Phase 16's gap: 0.0534
- Phase 16c's gap: 0.1261
- Phase 16d's gap: 0.6110

## Verdict

Decay is monotonic -- consistent with a genuine, continuously usable dial even under the new dedicated-heads architecture (confirms blending two independently-normalized, independently-trained embeddings still interpolates coherently, the thing this check was specifically built to verify rather than assume).

## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.925 | 0.835 | 0.709 | 0.569 | 0.444 | 0.354 | 0.297 | 0.264 | 0.252 | 0.249 |
| 0.1 | 0.925 | 1.000 | 0.894 | 0.756 | 0.603 | 0.467 | 0.369 | 0.308 | 0.273 | 0.258 | 0.254 |
| 0.2 | 0.835 | 0.894 | 1.000 | 0.843 | 0.675 | 0.525 | 0.411 | 0.339 | 0.299 | 0.281 | 0.276 |
| 0.3 | 0.709 | 0.756 | 0.843 | 1.000 | 0.807 | 0.634 | 0.497 | 0.406 | 0.354 | 0.330 | 0.321 |
| 0.4 | 0.569 | 0.603 | 0.675 | 0.807 | 1.000 | 0.797 | 0.631 | 0.517 | 0.447 | 0.413 | 0.398 |
| 0.5 | 0.444 | 0.467 | 0.525 | 0.634 | 0.797 | 1.000 | 0.800 | 0.658 | 0.567 | 0.519 | 0.497 |
| 0.6 | 0.354 | 0.369 | 0.411 | 0.497 | 0.631 | 0.800 | 1.000 | 0.828 | 0.717 | 0.653 | 0.620 |
| 0.7 | 0.297 | 0.308 | 0.339 | 0.406 | 0.517 | 0.658 | 0.828 | 1.000 | 0.865 | 0.786 | 0.744 |
| 0.8 | 0.264 | 0.273 | 0.299 | 0.354 | 0.447 | 0.567 | 0.717 | 0.865 | 1.000 | 0.904 | 0.852 |
| 0.9 | 0.252 | 0.258 | 0.281 | 0.330 | 0.413 | 0.519 | 0.653 | 0.786 | 0.904 | 1.000 | 0.934 |
| 1.0 | 0.249 | 0.254 | 0.276 | 0.321 | 0.398 | 0.497 | 0.620 | 0.744 | 0.852 | 0.934 | 1.000 |

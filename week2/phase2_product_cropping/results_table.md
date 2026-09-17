# Phase 2 Results: Product-Region Cropping vs Uncropped Baseline

All numbers below (baseline, Method A, Method B) are computed on the same 1854-product common evaluable set (see data/common_evaluable_set.md) so Hit Rate@K/Precision@K are directly comparable across variants -- see script docstring for why.

| Encoder | Variant | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|---|
| SigLIP | Uncropped (baseline) | 0.501 | 0.576 | 0.191 | 0.144 |
| SigLIP | Method A (rembg) | 0.508 | 0.574 | 0.199 | 0.150 |
| SigLIP | Method B (Grounding DINO) | 0.495 | 0.563 | 0.187 | 0.140 |
| FashionCLIP | Uncropped (baseline) | 0.464 | 0.538 | 0.179 | 0.136 |
| FashionCLIP | Method A (rembg) | 0.435 | 0.514 | 0.151 | 0.114 |
| FashionCLIP | Method B (Grounding DINO) | 0.440 | 0.498 | 0.164 | 0.122 |

## For reference: original phase 1b baseline (full 1,872-product pool, not the 1854-product common set)

| Encoder | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| SigLIP | 0.502 | 0.577 | 0.191 | 0.144 |
| FashionCLIP | 0.468 | 0.539 | 0.180 | 0.136 |

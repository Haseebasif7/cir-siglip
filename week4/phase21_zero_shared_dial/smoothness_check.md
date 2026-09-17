# Phase 21: Adjacent vs Distant Alpha Overlap -- Does Blending Two Independent Networks Stay Coherent?

Same method as phases 12d/16/16c/16d/17, full pairwise top-10 overlap matrix across all 11 alpha values, 500 queries. The complement and substitute heads here share ZERO parameters -- this is the direct test of whether alpha-blending two independently-trained embedding spaces still produces a coherent dial, or an abrupt jump/incoherent middle range (a version of phase 11's blending, already identified as not a real architectural contribution).

## Mean overlap as a function of |delta alpha|

| |delta alpha| | Mean top-10 overlap |
|---|---|
| 0.1 | 0.8184 |
| 0.2 | 0.6516 |
| 0.3 | 0.4995 |
| 0.4 | 0.3711 |
| 0.5 | 0.2702 |
| 0.6 | 0.1952 |
| 0.7 | 0.1436 |
| 0.8 | 0.1125 |
| 0.9 | 0.0946 |
| 1.0 | 0.0846 |

**Adjacent steps (delta=0.1): mean overlap = 0.8184**
**Most distant (delta=1.0): mean overlap = 0.0846**
**Gap (adjacent - distant): 0.7338**

## Comparison across the project's mechanisms

| Mechanism | Architecture | Gap |
|---|---|---|
| Phase 12d (substitute/complement) | shared trunk + additive correction | 0.4751 |
| Phase 16 (relevance/tail) | shared trunk + additive correction | 0.0534 |
| Phase 16c (relevance/tail) | shared trunk + additive correction | 0.1261 |
| Phase 16d (relevance/tail) | dedicated capacity, one shared layer | 0.6110 |
| Phase 17 (substitute/complement) | dedicated capacity, one shared layer | 0.6646 |
| **Phase 21 (substitute/complement)** | **ZERO shared parameters** | **0.7338** |

## Verdict

Decay is monotonic -- blending two fully independent networks' outputs (no shared parameters at all) STILL produces a genuine, smoothly usable dial: moving alpha a little changes retrieval a little, moving it a lot changes it a lot, with no abrupt jump or incoherent middle range. This answers this phase's central open question directly: the two independent embedding spaces are not unrelated in the way phase 11's separately-trained-and-blended models were -- both heads are functions of the SAME underlying frozen SigLIP embedding for every item, which plausibly keeps their 128-d output spaces geometrically compatible enough for a linear blend to interpolate coherently, even with zero shared training.

## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.912 | 0.816 | 0.698 | 0.561 | 0.419 | 0.288 | 0.190 | 0.133 | 0.101 | 0.085 |
| 0.1 | 0.912 | 1.000 | 0.891 | 0.764 | 0.620 | 0.464 | 0.317 | 0.208 | 0.142 | 0.107 | 0.088 |
| 0.2 | 0.816 | 0.891 | 1.000 | 0.858 | 0.703 | 0.530 | 0.366 | 0.244 | 0.164 | 0.122 | 0.098 |
| 0.3 | 0.698 | 0.764 | 0.858 | 1.000 | 0.823 | 0.637 | 0.452 | 0.306 | 0.207 | 0.154 | 0.121 |
| 0.4 | 0.561 | 0.620 | 0.703 | 0.823 | 1.000 | 0.788 | 0.580 | 0.403 | 0.275 | 0.205 | 0.161 |
| 0.5 | 0.419 | 0.464 | 0.530 | 0.637 | 0.788 | 1.000 | 0.762 | 0.554 | 0.392 | 0.292 | 0.229 |
| 0.6 | 0.288 | 0.317 | 0.366 | 0.452 | 0.580 | 0.762 | 1.000 | 0.755 | 0.559 | 0.426 | 0.334 |
| 0.7 | 0.190 | 0.208 | 0.244 | 0.306 | 0.403 | 0.554 | 0.755 | 1.000 | 0.768 | 0.600 | 0.476 |
| 0.8 | 0.133 | 0.142 | 0.164 | 0.207 | 0.275 | 0.392 | 0.559 | 0.768 | 1.000 | 0.802 | 0.651 |
| 0.9 | 0.101 | 0.107 | 0.122 | 0.154 | 0.205 | 0.292 | 0.426 | 0.600 | 0.802 | 1.000 | 0.825 |
| 1.0 | 0.085 | 0.088 | 0.098 | 0.121 | 0.161 | 0.229 | 0.334 | 0.476 | 0.651 | 0.825 | 1.000 |

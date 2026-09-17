# Phase 17: Adjacent vs Distant Alpha Overlap

Same method as phases 12d/16/16c/16d, full pairwise top-10 overlap matrix across all 11 alpha values, 500 queries.

## Mean overlap as a function of |delta alpha|

| |delta alpha| | Mean top-10 overlap |
|---|---|
| 0.1 | 0.8226 |
| 0.2 | 0.6610 |
| 0.3 | 0.5169 |
| 0.4 | 0.3959 |
| 0.5 | 0.3034 |
| 0.6 | 0.2372 |
| 0.7 | 0.1953 |
| 0.8 | 0.1722 |
| 0.9 | 0.1598 |
| 1.0 | 0.1580 |

**Adjacent steps (delta=0.1): mean overlap = 0.8226**
**Most distant (delta=1.0): mean overlap = 0.1580**
**Gap (adjacent - distant): 0.6646**

## Comparison across the project's mechanisms

| Mechanism | Architecture | Gap |
|---|---|---|
| Phase 12d (substitute/complement) | shared trunk + additive correction | 0.4751 |
| Phase 16 (relevance/tail) | shared trunk + additive correction | 0.0534 |
| Phase 16c (relevance/tail) | shared trunk + additive correction | 0.1261 |
| Phase 16d (relevance/tail) | dedicated capacity | 0.6110 |
| **Phase 17 (substitute/complement)** | **dedicated capacity** | **0.6646** |

## Verdict

Decay is monotonic -- consistent with a genuine, continuously usable dial under the dedicated-heads architecture, on the substitute/complement axis this time (not just the relevance/tail axis phase 16d tested).

## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.902 | 0.796 | 0.674 | 0.535 | 0.399 | 0.288 | 0.215 | 0.181 | 0.162 | 0.158 |
| 0.1 | 0.902 | 1.000 | 0.880 | 0.749 | 0.600 | 0.448 | 0.323 | 0.235 | 0.190 | 0.166 | 0.158 |
| 0.2 | 0.796 | 0.880 | 1.000 | 0.853 | 0.691 | 0.522 | 0.378 | 0.273 | 0.215 | 0.183 | 0.169 |
| 0.3 | 0.674 | 0.749 | 0.853 | 1.000 | 0.818 | 0.631 | 0.464 | 0.335 | 0.258 | 0.214 | 0.193 |
| 0.4 | 0.535 | 0.600 | 0.691 | 0.818 | 1.000 | 0.789 | 0.595 | 0.435 | 0.330 | 0.267 | 0.235 |
| 0.5 | 0.399 | 0.448 | 0.522 | 0.631 | 0.789 | 1.000 | 0.774 | 0.580 | 0.442 | 0.352 | 0.300 |
| 0.6 | 0.288 | 0.323 | 0.378 | 0.464 | 0.595 | 0.774 | 1.000 | 0.773 | 0.601 | 0.477 | 0.394 |
| 0.7 | 0.215 | 0.235 | 0.273 | 0.335 | 0.435 | 0.580 | 0.773 | 1.000 | 0.792 | 0.634 | 0.520 |
| 0.8 | 0.181 | 0.190 | 0.215 | 0.258 | 0.330 | 0.442 | 0.601 | 0.792 | 1.000 | 0.809 | 0.671 |
| 0.9 | 0.162 | 0.166 | 0.183 | 0.214 | 0.267 | 0.352 | 0.477 | 0.634 | 0.809 | 1.000 | 0.837 |
| 1.0 | 0.158 | 0.158 | 0.169 | 0.193 | 0.235 | 0.300 | 0.394 | 0.520 | 0.671 | 0.837 | 1.000 |

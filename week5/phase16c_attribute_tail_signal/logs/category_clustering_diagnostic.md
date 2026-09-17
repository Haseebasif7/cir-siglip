# Phase 16c: Category-Clustering Diagnostic (bonus, not a required step)

Tests the leading mechanistic hypothesis for the axis-check reversal directly: does tail-exposure mode's top-5 retrieval share the query's fine-grained category (phase 8's product_types.json breadcrumb) more often than relevance mode's does?

Queries with a known fine-grained type: 1732/1872.

| alpha | Mean same-category fraction @5 |
|---|---|
| 0.0 | 0.4410 |
| 0.1 | 0.4408 |
| 0.2 | 0.4413 |
| 0.3 | 0.4411 |
| 0.4 | 0.4419 |
| 0.5 | 0.4417 |
| 0.6 | 0.4406 |
| 0.7 | 0.4413 |
| 0.8 | 0.4416 |
| 0.9 | 0.4405 |
| 1.0 | 0.4410 |

**Tail-exposure (a=0.0): 0.4410 vs. Relevance (a=1.0): 0.4410**

Does not clearly confirm the same-category-clustering hypothesis -- the reversal likely has a different or additional mechanistic explanation, reported as such rather than forcing this explanation to fit.

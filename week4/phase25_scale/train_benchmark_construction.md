# Phase 25: Train-Split CIR Benchmark Construction

Built with the identical method as the validation/test benchmarks (candidate pools capped at 3,000/category, leave-one-out queries), pointed at `polyvore_raw/nondisjoint/train.json`. Used only for the per-epoch train-side Recall@10 diagnostic in phase 25's overfitting check (`overfitting_check.md`) -- never used as a training signal itself, and never compared against the actual test benchmark.

- Full construction: 46372 leave-one-out queries, 33000 pool slots across 11 categories.
- Subsampled to 5000 queries (seed=42) for per-epoch eval cost -- pools are kept at their full capped size, only the query set is subsampled.

| Category | Pool size | Full queries | Subsampled queries |
|---|---|---|---|
| tops | 3000 | 4053 | 447 |
| jewellery | 3000 | 4083 | 438 |
| bottoms | 3000 | 4302 | 460 |
| bags | 3000 | 4248 | 447 |
| shoes | 3000 | 4249 | 458 |
| hats | 3000 | 4592 | 526 |
| accessories | 3000 | 3829 | 442 |
| all-body | 3000 | 3700 | 395 |
| outerwear | 3000 | 3858 | 411 |
| sunglasses | 3000 | 5741 | 607 |
| scarves | 3000 | 3717 | 369 |


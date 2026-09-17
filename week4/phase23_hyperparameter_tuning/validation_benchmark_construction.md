# Phase 23, Step 1: Validation-Split CIR Benchmark Construction

Built with the exact same method as the test benchmark (`week4/phase12_controllable_modes/scripts/01_build_cir_benchmark.py`: candidate pools capped at 3,000 items/category via seed=42 sampling, leave-one-out queries kept only if their target landed inside the capped sample) -- the only change is the source split file, `polyvore_raw/nondisjoint/valid.json` instead of `test.json`. Kept in its own file (`data/cir_val_benchmark.json`), never merged with or substituted for the test benchmark.

## Coverage, validation split

- 5000 valid-split outfits total, 0 skipped (fewer than 2 qualifying items).
- 25132 unique valid-split items, 0 failed the qualifying check (missing image or unknown semantic_category).
- 4186 leave-one-out item-slots dropped (target fell outside its category's capped pool sample).
- **22595 leave-one-out queries kept**, across 11 category pools, 21171 total pool slots.

## Side-by-side against the test benchmark (already-validated construction)

| Metric | Validation benchmark | Test benchmark |
|---|---|---|
| Queries | 22595 | 29681 |
| Categories | 11 | 11 |
| Total pool slots | 21171 | 26494 |

## Per-category pool size and query count, both benchmarks

| Category | Val pool size | Val queries | Test pool size | Test queries |
|---|---|---|---|---|
| jewellery | 3000 | 3209 | 3000 | 3352 |
| bottoms | 2842 | 3060 | 3000 | 3361 |
| all-body | 1688 | 1730 | 3000 | 3182 |
| bags | 3000 | 3190 | 3000 | 3382 |
| outerwear | 1699 | 1754 | 3000 | 3244 |
| tops | 3000 | 3146 | 3000 | 3263 |
| shoes | 3000 | 3224 | 3000 | 3315 |
| sunglasses | 1224 | 1453 | 2202 | 2906 |
| accessories | 654 | 677 | 1254 | 1361 |
| hats | 645 | 710 | 1194 | 1401 |
| scarves | 419 | 442 | 844 | 914 |

## Composition check

Validation benchmark is 76.1% the size of the test benchmark by query count (22595 vs 29681), roughly in line with valid.json holding half as many outfits as test.json (5,000 vs 10,000). Per-category pool-size ratios range 49.6%-100.0% of the test benchmark's own pool sizes.

**Composition is proportionally consistent across categories** -- no category is disproportionately shrunk or missing relative to the others, so the validation benchmark's category mix is a reasonable, representative stand-in for the test benchmark's own mix, just built from fewer outfits overall.


# Phase 32, Step 5: Final Evaluation -- Test Benchmark, Once

3-seed ensemble (42, 1, 2), score-averaged, identified entirely through validation-benchmark comparisons (step 4: val R@10=0.2035, +4.8% over the best solo seed). Evaluated here, exactly once, on the actual test CIR benchmark.

## Full progression

| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |
|---|---|---|---|---|
| Phase 14b original (single config, un-invested) | 0.0588 | 0.1286 | 0.1809 | superseded baseline |
| Phase 31 single model (fully tuned, no ensemble) | 0.1799 | 0.3111 | 0.3844 | |
| **Phase 32, 3-seed partial ensemble** | **0.1897** | **0.3246** | **0.4019** | budget-constrained, 3 of a possible 10 seeds |
| Phase 28 text ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 | |

## Derived comparisons

- vs. phase 31's single model: +5.5% / +4.3% / +4.5% relative (beats it at every K).
- vs. phase 14b's original: +222.6% / +152.4% / +122.2% relative.
- vs. phase 28's own 10-model ensemble: 99.6% / 99.4% / 98.5% (still trails it at every K).

n_total=29681 n_skipped=0.


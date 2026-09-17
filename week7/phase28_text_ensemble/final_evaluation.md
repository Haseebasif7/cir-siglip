# Phase 28, Step 5: Final Evaluation -- Test Benchmark, Once

Best ensemble composition identified entirely through validation-benchmark comparisons (steps 3-4): size=10, seeds=[42, 1, 2, 3, 4, 5, 6, 7, 8, 9], validation Recall@10=0.2047. Evaluated here, exactly once, on the actual test CIR benchmark.

## Full progression

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Week 6 image-only, single tuned model | 0.1473 | 0.2684 | 0.3442 |
| Phase 27 text-only, single seed (corrected number, see phase27_notes.md) | 0.1656 | 0.2947 | 0.3733 |
| Phase 28 solo (seed=42, reused from phase 27, re-measured here for a like-for-like reference point) | 0.1656 | 0.2947 | 0.3733 |
| Week 6 image-only, 10-model ensemble | 0.1767 | 0.3054 | 0.3828 |
| **Phase 28 text-only, 10-model ensemble (seeds [42, 1, 2, 3, 4, 5, 6, 7, 8, 9])** | **0.1904** | **0.3267** | **0.4079** |

**The text-only ensemble beats week 6's image-only ensemble at every K.** Relative gain at K=10: +7.7% (single-seed signal from phase 27, corrected, predicted +12.4%).


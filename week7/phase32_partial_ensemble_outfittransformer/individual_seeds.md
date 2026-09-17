# Phase 32: Individual Seed Results (Validation Benchmark)

Each seed trained at phase 31's exact winning configuration (`input_mode=image_text, lr=1.5e-4,
batch_size=384, uniformity_weight=0.1, margin=0.2, selection_metric=recall10, patience=25,
max_epochs=100`), differing only in random seed.

| Seed | Val Recall@10 (Modal-reported) | Best epoch | Provenance |
|---|---|---|---|
| 42 | 0.1924 | 83 | reused from phase 31's `ot31_budget_check_full.pt`, zero additional cost |
| 1 | 0.1942 | 85 | trained this phase |
| 2 | 0.1929 | 84 | trained this phase |

Range across the 3 seeds: 0.1924-0.1942 (spread 0.0018, mean 0.1932, std 0.00074) -- a notably TIGHT
spread, tighter than either of this project's own prior ensembling phases at comparable scale (phase 26's
image-only 10 seeds: spread 0.0043, std 0.0014; phase 28's text-only 10 seeds: spread 0.0056, std 0.0016).
Only 3 seeds here vs. 10 there, so this isn't a like-for-like comparison of the full distribution, but the
seeds that did train landed unusually close together. See `phase32_notes.md` for what this suggests about
the expected ensembling gain.

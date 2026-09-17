# Phase 33: Individual Seed Results (Validation Benchmark)

Each seed trained at step 3's winning configuration (`input_mode=image_text, lr=1e-4, batch_size=48,
selection_metric=recall10, patience=5, max_epochs=40`), differing only in random seed.

| Seed | Val Recall@10 (Modal-reported) | Best epoch | Epochs run |
|---|---|---|---|
| 42 | 0.1075 | 20 | 26 |
| 1 | 0.1008 | 16 | 22 |
| 2 | 0.1028 | 14 | 20 |

Range across the 3 seeds: 0.1008-0.1075 (spread 0.0067, mean 0.1037, std 0.0027). Wider spread than phase
32's OutfitTransformer seeds (spread 0.0018, std 0.00074), and closer to this project's own model's typical
seed variance (phase 26/28: std 0.0014-0.0016) -- a plausible sign of more per-query disagreement between
seeds, worth checking against the ensembling gain actually measured (see `phase33_notes.md`).

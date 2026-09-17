# Phase 36: Measured Inference Cost (full test-benchmark evaluation, as built)

Machine: local M4 (Apple Silicon). 3 full passes per system, median reported. Each pass = load checkpoints, precompute candidate embeddings, embed and score all 29681 test queries against their category pools (3,000 candidates or fewer), rank. Catalog: 251008 items. Device use differs by system exactly as in each phase's own evaluator and is stated per row -- this is the cost of the pipelines that produced the paper's numbers, not a device-controlled benchmark (see the CPU-only section below for that).

| System | members | trainable params (total) | precompute (s) | query + score (s) | total (s) | total per member (s) | per query (ms) | peak RSS (MB) | device split |
|---|---|---|---|---|---|---|---|---|---|
| `ours_solo` | 1 | 1,705,088 | 1.2 | 0.9 | 2.3 | 2.3 | 0.03 | 2923 | candidate projection: torch/mps; query pooling + cosine + ranking: NumPy/CPU (as the phase 28 evaluator) |
| `ours_ens` | 10 | 17,050,880 | 10.6 | 5.4 | 16.0 | 1.6 | 0.18 | 2923 | candidate projection: torch/mps; query pooling + cosine + ranking: NumPy/CPU (as the phase 28 evaluator) |
| `ot_solo` | 1 | 998,144 | 1.5 | 3.4 | 4.9 | 4.9 | 0.11 | 2923 | candidate embed_item_alone + query set-encoder forward: torch/mps; cosine + ranking: NumPy/CPU (as phase 32) |
| `ot_ens` | 3 | 2,994,432 | 4.5 | 9.7 | 14.2 | 4.7 | 0.33 | 2927 | candidate embed_item_alone + query set-encoder forward: torch/mps; cosine + ranking: NumPy/CPU (as phase 32) |
| `csa_ens` | 3 | 301,455 | 0.0 | 26.9 | 27.0 | 9.0 | 0.91 | 2928 | everything (pool candidate tensors, conditioned context embeddings, distances, ranking): torch/mps (as phase 34) |

For CSA-Net, candidate embeddings are built per category pool inside the query loop (as its evaluator does); the time spent on that is included in `query + score` and separately measured at 0.4 s (median).

## Recorded training cost (from each phase's own JSON; different hardware, NOT directly comparable across rows)

| System | seed | wall-clock (s) | device | where recorded |
|---|---|---|---|---|
| ours | 42 | 626 | cuda | Modal GPU (phase 28) |
| ours | 1 | 623 | cuda | Modal GPU (phase 28) |
| ours | 2 | 625 | cuda | Modal GPU (phase 28) |
| ours | 3 | 621 | cuda | Modal GPU (phase 28) |
| ours | 4 | 624 | cuda | Modal GPU (phase 28) |
| ours | 5 | 642 | cuda | Modal GPU (phase 28) |
| ours | 6 | 645 | cuda | Modal GPU (phase 28) |
| ours | 7 | 631 | cuda | Modal GPU (phase 28) |
| ours | 8 | 612 | cuda | Modal GPU (phase 28) |
| ours | 9 | 616 | cuda | Modal GPU (phase 28) |
| ot | 1 | 2400 | cuda | Modal GPU (phase 32) |
| ot | 2 | 2399 | cuda | Modal GPU (phase 32) |
| ot | 42 | 2418 | cuda | Modal GPU (phase 31, reused as seed 42) |
| csa | 1 | 3560 | mps | local M4 MPS (phase 34) |
| csa | 2 | 2517 | mps | local M4 MPS (phase 34) |
| csa | 42 | 2231 | mps | local M4 MPS (phase 34 gate run) |

Summed over the seeds actually used in each final ensemble: ours: 104 min (6266 s), ot: 120 min (7216 s), csa: 138 min (8308 s). Ours and OutfitTransformer trained on Modal cloud GPUs; CSA-Net trained on the laptop's MPS -- so wall-clock is only comparable within a row group, not across.

Interpretation in `phase36_notes.md`.

## Device-controlled comparison: every system forced onto CPU

3 passes, median. Same machine, no MPS use anywhere.

| System | members | precompute (s) | query + score (s) | total (s) | total per member (s) | per query (ms) |
|---|---|---|---|---|---|---|
| `ours_solo` | 1 | 1.3 | 0.9 | 2.2 | 2.2 | 0.03 |
| `ours_ens` | 10 | 11.6 | 5.2 | 16.9 | 1.7 | 0.18 |
| `ot_solo` | 1 | 3.6 | 7.2 | 10.7 | 10.7 | 0.24 |
| `ot_ens` | 3 | 11.0 | 21.7 | 33.0 | 11.0 | 0.73 |
| `csa_ens` | 3 | 0.0 | 38.5 | 38.5 | 12.8 | 1.30 |

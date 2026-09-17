# Phase 13: CSA-Net Reproduction -- Results Table

Evaluated on this project's own CIR harness (`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged since phase 12): 26494 pool slots, 29681 queries total in the benchmark; 29681 scored for CSA-Net (0 skipped -- see `phase13_notes.md` for why, if nonzero).

## This project's reproduction vs. CSA-Net's own published numbers

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| **This reproduction (own harness)** | 0.0284 | 0.0667 | 0.0960 |
| CSA-Net published (paper Table 4, Polyvore Outfits non-disjoint) | 0.0827 | 0.1567 | 0.2091 |

Note: the published numbers come from the paper's OWN candidate-pool construction (27/153 fine-grained categories, each capped at 3,000 images) which differs from this project's own harness (11 broad semantic categories, same 3,000 cap) -- so this is a similar-protocol sanity check, not a byte-for-byte identical evaluation. See `phase13_notes.md` for what a close/far match does and doesn't establish.

## Against this project's own configurations (identical harness, identical benchmark)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Raw SigLIP (alone) | 0.0553 | 0.1067 | 0.1437 |
| Phase 9 Model A (alone) | 0.1317 | 0.2464 | 0.3216 |
| Phase 12c: Substitute mode | 0.0667 | 0.1307 | 0.1734 |
| Phase 12c: Complement mode | 0.0971 | 0.1875 | 0.2471 |
| Phase 12c: Blend (0.5) | 0.0893 | 0.1695 | 0.2255 |
| **Phase 13: CSA-Net reproduction** | **0.0284** | **0.0667** | **0.0960** |


# Phase 13b: CSA-Net's Subspace Attention on Frozen SigLIP -- Results Table

Evaluated on this project's own CIR harness (`week4/phase12_controllable_modes/data/cir_benchmark.json`,
unchanged since phase 12): 26,494 pool slots, 29,681 queries, all scored
(0 skipped). Trained to genuine convergence (full 40-epoch LR-decay schedule
completed, best checkpoint from epoch 38 -- see `training_log.md`), **not**
budget-truncated like phase 13's full reproduction.

**Labeling, kept distinct per this phase's own brief:**
- **"This configuration"** = CSA-Net's category-pair subspace attention
  mechanism, trained on top of a FROZEN SigLIP backbone (this phase).
- **"Phase 13 (undertrained)"** = CSA-Net's own ResNet18 architecture,
  fine-tuned end-to-end, stopped early at 5/20 epochs by a real budget
  constraint -- NOT comparable as a finished result, included only as a
  before/after reference point.
- **"CSA-Net published"** = the paper's own numbers, their own backbone
  (ResNet18) and their own candidate-pool protocol (27/153 fine-grained
  categories) -- not a byte-for-byte matched evaluation, a similar-protocol
  reference only.

## This configuration vs. the other two CSA-Net numbers

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| **This configuration (CSA-Net mechanism, frozen SigLIP)** | **0.0725** | **0.1393** | **0.1844** |
| CSA-Net published (paper Table 4, Polyvore Outfits non-disjoint) | 0.0827 | 0.1567 | 0.2091 |
| Phase 13 (undertrained ResNet18 reproduction, NOT comparable) | 0.0284 | 0.0667 | 0.0960 |

This configuration reaches **87.7% / 88.9% / 88.2%** of CSA-Net's own
published Recall@10/30/50 respectively -- consistently close across all
three K values, despite using a different backbone than the paper and a
different (this project's own) candidate-pool protocol. See
`phase13b_notes.md` for what this closeness does and doesn't establish.

## Against this project's own configurations (identical harness, identical benchmark, identical backbone -- the actually meaningful comparison)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Raw SigLIP (alone, no training at all) | 0.0553 | 0.1067 | 0.1437 |
| Phase 9 Model A (alone) | 0.1317 | 0.2464 | 0.3216 |
| Phase 12c: Substitute mode | 0.0667 | 0.1307 | 0.1734 |
| **Phase 13b: CSA-Net mechanism (this phase)** | **0.0725** | **0.1393** | **0.1844** |
| Phase 12c: Blend (0.5) | 0.0893 | 0.1695 | 0.2255 |
| Phase 12c: Complement mode | 0.0971 | 0.1875 | 0.2471 |

(Table rows ordered by Recall@10, ascending, so this phase's row's position
relative to this project's own configurations is easy to read at a glance.)

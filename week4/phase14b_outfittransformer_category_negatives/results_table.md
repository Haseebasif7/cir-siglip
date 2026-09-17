# Phase 14b: Results Table

Same benchmark used throughout this project since phase 12
(`week4/phase12_controllable_modes/data/cir_benchmark.json`): 29,681
queries, 0 skipped, every row below evaluated on the identical protocol.

**Labeling, kept consistent with phase 14's own convention:**
- **"mine"** = trained and evaluated in this project, on this exact
  harness.
- **"published"** = the paper's own reported number, different backbone
  and candidate-pool construction, cited as directional context only, not
  a matched-protocol claim.

## Full comparison, ordered by Recall@10 ascending

| Configuration | Source | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|---|
| Phase 14b, run 1: OutfitTransformer, mined same-category negatives | mine | 0.0051 | 0.0149 | 0.0235 |
| Phase 14 (original): OutfitTransformer, unrestricted in-batch negatives | mine | 0.0201 | 0.0588 | 0.0911 |
| Raw SigLIP (alone, no training) | mine | 0.0553 | 0.1067 | 0.1437 |
| **Phase 14b, run 2: OutfitTransformer, random same-category negatives** | **mine** | **0.0588** | **0.1286** | **0.1809** |
| Phase 13b: CSA-Net mechanism, frozen SigLIP backbone | mine | 0.0725 | 0.1393 | 0.1844 |
| OutfitTransformer published (paper, Polyvore Outfits) | published | 0.0958 | 0.1796 | 0.2198 |

## Phase 14b's two runs against phase 14's original broken baseline

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Run 1 (mined negatives) vs. phase 14 | 0.25x | 0.25x | 0.26x |
| **Run 2 (random negatives) vs. phase 14** | **2.93x** | **2.19x** | **1.99x** |

Run 1 made the category-match problem worse in a new way: the negatives
became too hard for this small transformer's triplet-margin loss to
resolve at all (see `training_log.md` -- the margin stayed violated by
~0.30 the entire 100-epoch run). Run 2, drawing from the same fixed
category-restricted pool but at random rather than by SigLIP-nearest-neighbor
mining, is the version that actually delivers the fix's intended benefit.

## Run 2 against the literature and this project's own configurations

| Comparison | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Run 2 as % of OutfitTransformer published | 61.4% | 71.6% | 82.3% |
| Run 2 as % of CSA-Net reproduction (frozen SigLIP, this project) | 81.1% | 92.3% | 98.1% |
| Run 2 as % of raw SigLIP | 106.3% | 120.5% | 125.9% |

Run 2 now beats raw SigLIP at every K (the first OutfitTransformer-mechanism
configuration in this project to do so), and lands within 2-19 percentage
points of CSA-Net's own frozen-SigLIP reproduction depending on K, closing
most of the gap to it. It still falls short of OutfitTransformer's own
published numbers, and the ratio to published numbers climbs with K
(61% to 72% to 82%) rather than sitting flat -- unlike CSA-Net's
reproduction, which landed at a consistent ~88% across all three K values.
See `phase14b_notes.md` for the honest verdict against that bar.

# Phase 14b: Training Log

## Step 0: negative sampling verification (before any training)

See `negative_sampling_fix_summary.md`. Category-match rate went from 13.1%
(phase 14's diagnosed problem) to 100.0% (192,000 sampled negatives
checked, 0 short rows). Gate passed, proceeded to training.

## Smoke test: collapse check and D_pos/D_neg sanity

5 epochs on a 2,000-outfit subset (batch=96, `use_uniformity=True` from the
start -- phase 14's own smoke test already established this architecture
needs it to avoid direction collapse, no reason to re-litigate that with
`use_uniformity=False` here since only the negative source changed, not the
collapse-prone parts of the architecture):

```
epoch 0: train_loss=-1.1190 val_loss=-1.0398 val_D_pos=0.8890 val_D_neg=0.7334
epoch 1: train_loss=-2.5295 val_loss=-2.5104 val_D_pos=1.2817 val_D_neg=1.0884
epoch 2: train_loss=-2.9768 val_loss=-2.7552 val_D_pos=1.3293 val_D_neg=1.1487
epoch 3: train_loss=-3.0765 val_loss=-2.7549 val_D_pos=1.3331 val_D_neg=1.1595
epoch 4: train_loss=-3.0893 val_loss=-2.7756 val_D_pos=1.3329 val_D_neg=1.1606
```

Loss goes negative from the start -- expected, not a bug, same pattern
phase 14 itself shows throughout its own `training_log.md` (its final
val_loss was -3.5050): the uniformity term (`log(mean(exp(-t*sq_dist)))`)
is naturally a large negative number once embeddings are well spread out on
the hypersphere, and it's summed directly into the reported loss alongside
the (always non-negative) triplet term. The real ranking diagnostic is the
D_pos/D_neg gap, not the raw loss sign.

No dead gradients, no crash, 13.7s for 5 epochs on the subset. Ran fine on
the first attempt (this phase reuses phase 14's own `math.ceil` steps-per-epoch
fix for `OneCycleLR`, so that specific bug phase 14 hit doesn't recur here).

## Timing test: full dataset, 1 epoch

38.2s for 1 full epoch (53,306 training outfits, batch=96,
`use_uniformity=True`) -- slower than phase 14's 24.5s/epoch, as expected:
each step now also runs NUM_NEGATIVES=10 extra items per anchor through the
transformer (as their own length-1 `embed_item_alone` sequences) on top of
the context and target items phase 14 already processed. Projected 100
epochs at ~64 minutes. Local (MPS) is still clearly practical -- no Modal
GPU spending triggered.

## Full training run 1: mined same-category negatives

Launched locally (MPS) under `caffeinate -is`, same convention phase 14
used to avoid a laptop-sleep interruption. Same hyperparameters as phase 14
(`lr=2e-5`, `batch_size=96`, `margin=0.3`, `max_epochs=100`, `patience=8`,
`use_uniformity=True` from the start) -- only the negative source changed:
negatives drawn via `negative_mode="mined"` (CSA-Net's own mined
same-category candidate list, `week4/phase13_csa_net_baseline/data/negative_candidates.json`,
top-20 SigLIP-nearest same-category neighbors per item, capped at 0.97
similarity).

- **Total wall time: 4793.3s (~79.9 min)** for 100 epochs, ran to the full
  schedule, no early stop triggered (patience=8 never hit -- val_loss kept
  inching down every few epochs to the very end, epoch 92's checkpoint was
  the last "new best").
- **Best checkpoint: epoch 92** (`val_loss=-3.5281`).
- Final-model collapse check on the trained candidate embeddings (all
  251,008 catalog items, same check phase 14 ran): mean pairwise cosine
  similarity 0.0015 (N=2000 sample), norms all exactly 1.0. **No collapse.**

### The real diagnostic: the triplet-margin term itself, not the reported loss

The reported training/val loss (final val_loss=-3.5278) looks like a clean
convergence, matching phase 14's own final val_loss of -3.5050 almost
exactly. But that number is dominated by the uniformity regularizer (a
large negative term whenever embeddings are well spread out, unrelated to
ranking quality) -- I checked the ACTUAL triplet-margin component
(`max(0, D_pos - D_neg + margin)`, the only part of the loss that reflects
whether the model ranks a real target above its hardest negative) directly,
epoch by epoch, from `models/training_curves.json`:

| Epoch | val_D_pos | val_D_neg | pos - neg + margin (unresolved if > 0) |
|---|---|---|---|
| 0 | 1.2727 | 1.0857 | 0.4869 |
| 10 | 1.4109 | 1.3791 | 0.3318 |
| 20 | 1.4144 | 1.4027 | 0.3117 |
| 30 | 1.4110 | 1.4005 | 0.3105 |
| 40 | 1.4142 | 1.4051 | 0.3091 |
| 50 | 1.4143 | 1.4067 | 0.3076 |
| 60 | 1.4154 | 1.4087 | 0.3067 |
| 70 | 1.4153 | 1.4095 | 0.3058 |
| 80 | 1.4155 | 1.4098 | 0.3057 |
| 90 | 1.4154 | 1.4099 | 0.3054 |
| 99 (final) | 1.4154 | 1.4100 | 0.3054 |

**The triplet-margin term never resolves -- it drops fast in the first ~20
epochs (0.487 to 0.312) and then plateaus at ~0.305-0.306 for the remaining
80 epochs, never getting close to 0.** `val_D_pos` stays consistently
LARGER than `val_D_neg` the entire run: the model never learns to place the
true target closer than its hardest same-category mined negative. Both
distances converge toward ~1.414 (=sqrt(2), the expected Euclidean distance
between two uncorrelated unit vectors) -- consistent with the model mostly
using its capacity to satisfy the uniformity term (which it CAN solve,
since spreading embeddings apart is easy) while the actual ranking
objective against these specific negatives stays essentially unsolved.

### CIR benchmark result: worse than phase 14's broken baseline

`Recall@10=0.0051 Recall@30=0.0149 Recall@50=0.0235` (29,681 queries, 0
skipped) -- **roughly a quarter of phase 14's own already-broken
Recall@10=0.0201.** See `results_table.md` for the full comparison.

### Diagnosis: negative difficulty, not the category restriction itself

The category-match rate fix worked exactly as verified in step 0 (100%).
But the specific negatives it supplies -- SigLIP-mined same-category NEAR
NEIGHBORS -- are much harder to discriminate from the true target than
phase 14's arbitrary in-batch negatives (any other outfit's target,
unrestricted by category OR visual similarity) ever were. A small
4-layer/8-head transformer with a single triplet-margin comparison appears
unable to solve that harder discrimination task at all here: the loss
plateaus with the margin violated by ~0.3 for 80 straight epochs, not a
partial improvement that ran out of time.

This directly parallels a finding already established earlier in this
project (cited in the phase 16 plan, `week5/phase16_relevance_tail_dial/`):
phases 7-9 found that mined HARD negatives underperform plain random
negatives for this project's embedding-based retrieval losses. This is the
same mechanism resurfacing in a different loss family (triplet-margin vs.
MNRL/InfoNCE) and a different phase.

**Ran an ablation before writing this up as final** (see
`02b_train_full_random_negatives.py`): identical setup, but
`negative_mode="random"` -- negatives still drawn from the same category
(same fix, same category-restriction property), but uniformly at random
from the category's item pool rather than from the SigLIP-mined
nearest-neighbor list. This isolates whether the category restriction
itself is sound (as CSA-Net's own precedent, and phase 13's own reproduction
result, both suggest it should be) once it isn't compounded with maximal
negative difficulty.

## Full training run 2: random same-category negatives (ablation)

Same setup as run 1 in every respect except `negative_mode="random"`
(negatives drawn uniformly from the target's own category pool, skipping
the SigLIP-mined nearest-neighbor list entirely -- see
`02b_train_full_random_negatives.py` and `train_core.py`'s
`_sample_negatives`).

- **Total wall time: 11404.4s (~190.1 min)**, noticeably slower per epoch
  than run 1 (~124s/epoch average vs run 1's ~48s/epoch) -- the random-mode
  path filters the full category item list (up to 51,132 items for
  "shoes") on every sampled negative, an O(category size) operation per
  sample instead of run 1's O(20) mined-candidate-list sampling. A real,
  identified inefficiency, not investigated further since it didn't block
  completion.
- **Early stopping fired at epoch 91** (`patience=8`, last improvement at
  epoch 83 -- 83+8=91, exact match, a clean stop, not a crash or timeout).
- **Best checkpoint: epoch 83** (`val_loss=-3.5201`).
- Final-model collapse check (same as run 1, all 251,008 catalog items):
  mean pairwise cosine similarity 0.0024 (N=2000 sample), norms all 1.0.
  **No collapse.**

### D_pos/D_neg trajectory -- looked almost identical to run 1

| Epoch | val_D_pos | val_D_neg | pos - neg + margin |
|---|---|---|---|
| 0 | 1.2559 | 1.0785 | 0.4775 |
| 10 | 1.4067 | 1.3737 | 0.3329 |
| 20 | 1.4105 | 1.3995 | 0.3111 |
| 30 | 1.4097 | 1.4023 | 0.3074 |
| 40 | 1.4078 | 1.4027 | 0.3051 |
| 50 | 1.4065 | 1.4024 | 0.3040 |
| 60 | 1.4066 | 1.4035 | 0.3031 |
| 70 | 1.4061 | 1.4033 | 0.3028 |
| 80 | 1.4060 | 1.4035 | 0.3025 |
| 83 (best) | 1.4058 | 1.4033 | 0.3025 |
| 90 | 1.4060 | 1.4035 | 0.3025 |

Watching this trajectory live during training, it tracked run 1's almost
exactly at every checkpoint (e.g. epoch 20: run 1's `pos-neg+margin=0.3117`
vs run 2's `0.3111`; epoch 80: run 1's `0.3057` vs run 2's `0.3025`) -- both
runs plateau with the triplet-margin term unresolved by essentially the
same amount, ~0.30. On this diagnostic alone, the two runs look nearly
indistinguishable.

### CIR benchmark result: the diagnostic was misleading -- Recall@K tells a very different story

`Recall@10=0.0588 Recall@30=0.1286 Recall@50=0.1809` (29,681 queries, 0
skipped).

**This is a real, substantial improvement over BOTH run 1 (mined
negatives) and phase 14's own original broken baseline**, even though the
internal D_pos/D_neg diagnostic gave almost no hint of a difference between
run 1 and run 2:

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Run 2 vs. phase 14 (broken baseline) | 2.93x | 2.19x | 1.99x |
| Run 2 vs. run 1 (mined negatives) | 11.5x | 8.6x | 7.7x |

**Corrected diagnosis, replacing my earlier in-progress read of this
ablation**: while the run was in progress, the matching D_pos/D_neg
plateaus looked like they ruled out negative difficulty as the cause and
pointed to the category restriction itself being the bottleneck. The
actual Recall@K result contradicts that reading directly -- negative
difficulty clearly matters enormously (an 8-11x swing in real retrieval
quality), the internal training diagnostic just doesn't reflect it. This
is the same lesson phase 14 already learned once, from the opposite
direction (`phase14_notes.md`: "the training diagnostic's near-zero gap is
consistent with the model satisfying the aggregate training objective well
without being consistent with the model producing a fine-grained, reliable
per-query ranking") -- reconfirmed here, and worth stating plainly: this
project's own D_pos/D_neg gap, while useful for catching outright collapse
or dead gradients, is not a reliable predictor of real Recall@K for this
mechanism, in either direction.

See `results_table.md` for the full comparison table and
`phase14b_notes.md` for the honest verdict against the brief's bar.

# Phase 15b: Training Log

## Step 3 safeguards (checked directly, not assumed to carry over from phase 15)

Full detail in `training_log_smoke_test.md`. Summary:

- **Dead-gradient check** (untrained model, one real 96-outfit batch): every
  shared parameter, including `attn_net`'s alpha-input weights, received a
  nonzero gradient (smallest: `attn_net.0.weight` at 0.000231 -- small but
  real, same as phase 15's own init behavior, not a dead-gradient trap).
- **Collapse check**: mean pairwise candidate-embedding cosine similarity
  0.7512 without the uniformity regularizer vs. 0.3942 with it (5 epochs,
  2,000-outfit subset) -- consistent with phase 13/13b/14/15's recurring
  direction-collapse failure mode, fixed the same way. Uniformity
  regularizer (weight=1.0) enabled from the start of the real run.

## Timing

Local CPU, `caffeinate -is`, no Modal spending needed or used. A direct
10-step timing check before the real run measured 0.432s/step -> ~4.0
min/epoch projected -- essentially identical to phase 15's own per-step cost
(0.410s/step), confirming the dual fixed-alpha forward pass does NOT
meaningfully increase training time: the total number of forward passes per
step is unchanged from phase 15 (complement's calls now fixed at alpha=0,
substitute's calls now fixed at alpha=1, rather than both sharing one
sampled alpha) -- no doubling. The real run's measured pace was ~4.7
min/epoch (some run-to-run variance from other machine load), close to the
timing check's projection.

## The real training run

Ran the full 60-epoch schedule to completion (`02_train_full.py`) -- early
stopping (patience=8) never triggered; LR had decayed to ~1.5e-7 by the
final epoch. Best checkpoint: epoch 59 (last epoch), best `val_loss=-2.9798`.

| Epoch | train_loss | val_loss | val_D_pos | val_D_neg | gap | val_comp | val_sub |
|---|---|---|---|---|---|---|---|
| 0 | -1.8034 | -2.2269 | 1.9727 | 1.8915 | 0.0812 | 0.3812 | 0.0628 |
| 9 | -2.6347 | -2.6068 | 1.9480 | 1.8780 | 0.0700 | 0.3702 | 0.0413 |
| 19 | -2.8368 | -2.8039 | 1.8816 | 1.7963 | 0.0853 | 0.3854 | 0.0320 |
| 29 | -2.9341 | -2.8974 | 1.8572 | 1.7653 | 0.0919 | 0.3921 | 0.0294 |
| 39 | -2.9810 | -2.9453 | 1.8471 | 1.7479 | 0.0992 | 0.3994 | 0.0279 |
| 49 | -3.0023 | -2.9638 | 1.8502 | 1.7478 | 0.1024 | 0.4026 | 0.0276 |
| **59 (best)** | -3.0117 | -2.9673 (best epoch: -2.9798) | 1.8492 | 1.7487 | **0.1005** | 0.4007 | 0.0277 |

The D_pos/D_neg gap (computed at alpha=0, the complement pathway) plateaus
around ~0.09-0.10 -- somewhat WIDER than phase 15's own plateau (~0.08 for
both continuous and discrete), and val_comp is also somewhat higher
throughout (~0.40 vs. phase 15's ~0.38) -- the complement pathway's own
training diagnostic looks slightly less converged here than in phase 15,
despite (see `results_table.md`) its actual Recall@K coming out AT LEAST AS
GOOD as phase 15's discrete complement endpoint. This is yet another
instance of this project's repeated lesson (phases 14, 15) that a training
diagnostic's own value does not reliably predict downstream retrieval
quality -- checked here again rather than assumed.

## Post-training verification (re-checked directly on the final model, not assumed from the smoke test)

- **Attention-weight-shift probe** (`model.attention_weight_shift`, all 121
  category pairs): **mean L1 shift = 1.9955** (vs. phase 15's continuous
  0.188 and discrete 0.486, and an untrained baseline of ~0.041) -- **at
  essentially the theoretical maximum of 2.0 for a 5-way softmax pair, and
  uniformly so across every single category pair** (min 1.9545, max
  1.9998 -- the tightest spread of any configuration checked in this
  project). This is decisive, full-scale confirmation that removing alpha
  from the loss envelope resolved the double-duty confound: `attn_net` now
  produces a near-maximally distinct attention distribution at alpha=0 vs.
  alpha=1 for literally every category pair, not just some.
- **Adjacent-vs-distant smoothness check** (`04_smoothness_check.py`,
  identical methodology to phase 12d/15): gap = **0.6958**, verdict
  **"Real, smooth relationship confirmed"** -- clears phase 12d's own
  0.15 threshold by more than 4.6x, and exceeds even phase 12c/12d's own
  simple additive mechanism's gap (0.475) by 46%. The strongest
  smoothness result of any conditioning mechanism evaluated in this
  project's history.
- Full alpha-sweep Recall@K and the step-2 stop-condition evaluation:
  see `results_table.md` and `phase15b_notes.md`.

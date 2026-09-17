# Phase 15: Training Log

## Step 3/4 safeguards: see `architecture_notes.md` and `training_log_smoke_test.md`

Collapse check, conditioning-consistency check, and the attention-weight-shift
probe were all run as part of the smoke test (`02_smoke_test.py`) BEFORE the
real training runs below -- see `architecture_notes.md`'s "Named risk" and
"Collapse check" sections for the full numbers. Summary: uniformity
regularizer (weight=1.0) is needed and enabled from the start (collapse
0.7565 without it vs. 0.4039 with it, on a small smoke subset); the
attention-weight-shift probe showed no clear signal yet at smoke-test scale
(0.024-0.028 vs. 0.041 untrained baseline) -- re-checked on the real trained
checkpoints below before any claim is made about whether the conditioning
mechanism is actually realized.

## Step 5: local vs. Modal decision

### Training cost (step 4b's timing requirement)

A direct 20-step timing comparison (`TrainState` + `compute_batch_loss`,
fixed batch=96, alpha=0.5) was run on both devices before committing to a
device for the real runs:

| Device | s/step | Projected 1 epoch (555 steps) |
|---|---|---|
| MPS | 1.352 | 12.5 min |
| CPU | 0.410 | 3.8 min |

**CPU is ~3.3x faster than MPS for this workload.** This workload is
dominated by many small per-item tensor ops (the per-context-item complement
loop, the per-negative loop, the per-neighbor substitute loop -- all
individually tiny tensors), where MPS's per-op dispatch overhead outweighs
its raw compute advantage. All phase 15 scripts use `device="cpu"`, not the
MPS-first pattern phases 13b/14 defaulted to (which was correct for THEIR
workloads -- larger batched matrix ops with less Python-loop granularity).
This is consistent with, not a break from, phase 13b's own lesson ("swap the
expensive component for what's actually being tested" -- here the expensive
component is per-op dispatch overhead, not compute).

At ~3.8 min/epoch on CPU, a 60-epoch run (chosen to match phase 13b's own
budget, which converged around epoch 38 of 40) projects to **~3.8 hours
per run**, well within local budget for both the continuous and discrete
runs -- **no Modal GPU spending needed or used**, matching phase 13b/14's
precedent that this project's frozen-backbone mechanisms rarely need it.

### Eval sweep cost (step 4b's second timing requirement)

The full step-6 sweep (11 alpha points x 2 checkpoints x the diagnostic
suite) does per-context-item, per-query forward passes -- NOT a cached flat
lookup like phase 12d's simpler mechanism -- so it was timed directly before
running, per the plan's explicit caution that this could quietly dominate
the phase's wall-clock budget:

- 1 alpha point (includes one-time setup: precomputing `model.encode_feature`
  for the full 251,008-item catalog + raw-SigLIP reference top-K for 500
  overlap-diagnostic queries): 25.1s
- 2 alpha points: 44.4s
- **Marginal cost per additional alpha point: ~19.3s. One-time setup: ~5.8s.**
- **Projected full 11-point sweep: ~218s (~3.6 min) per checkpoint, ~7.3 min
  for both checkpoints.**

This is cheap -- no Modal spending needed for evaluation either, and the
eval sweep does NOT dominate the phase's wall-clock budget (training does,
at ~3.8 hours/run).

## Real training runs

Both trained locally on CPU under `caffeinate -is` (no Modal spending used
or needed for either run). D_pos/D_neg/comp/sub monitored continuously
throughout (not just at the end), full logs in
`logs/train_full_continuous.log` / `logs/train_discrete.log`.

### Continuous-alpha run (`04_train_full.py`)

Ran the full 60-epoch schedule to completion (early-stopping patience=8
never triggered; LR had decayed to ~1.5e-7 by the final epoch). Best
checkpoint: epoch 59 (last epoch, `val_loss=-3.1774`).

| Epoch | train_loss | val_loss | val_D_pos | val_D_neg | gap | val_comp | val_sub |
|---|---|---|---|---|---|---|---|
| 0 | -2.4097 | -2.7181 | 1.9432 | 1.8535 | 0.0897 | 0.3898 | 0.0915 |
| 9 | -3.0071 | -2.9881 | 1.9327 | 1.8583 | 0.0744 | 0.3745 | 0.0570 |
| 19 | -3.0741 | -3.0362 | 1.9309 | 1.8552 | 0.0757 | 0.3757 | 0.0517 |
| 29 | -3.1265 | -3.0873 | 1.9219 | 1.8480 | 0.0739 | 0.3740 | 0.0466 |
| 39 | -3.1724 | -3.1406 | 1.9016 | 1.8213 | 0.0803 | 0.3804 | 0.0426 |
| 49 | -3.1993 | -3.1697 | 1.8979 | 1.8164 | 0.0815 | 0.3816 | 0.0398 |
| **59 (best)** | -3.2083 | -3.1774 | 1.8945 | 1.8140 | **0.0805** | 0.3805 | 0.0391 |

Gap (D_pos - D_neg) plateaus around ~0.08 from roughly epoch 20 onward --
comparable to phase 13b's own plateau (0.082) on the identical backbone
without alpha-conditioning, i.e. adding the conditioning mechanism did not
noticeably change this diagnostic's convergence quality. `val_sub` (the
substitute KL loss) shrinks steadily throughout (0.0915 -> 0.0391), while
`val_comp` stays roughly flat (~0.37-0.38) -- consistent with the loss
envelope giving each term real, ongoing gradient signal across training
(not just at the two chosen alpha values), even though alpha is sampled
continuously.

### Discrete-alpha run (`05_train_discrete.py`)

Early-stopped at epoch 57 (8 epochs with no improvement); best checkpoint:
epoch 49 (`val_loss=-3.1927`).

| Epoch | train_loss | val_loss | val_D_pos | val_D_neg | gap | val_comp | val_sub |
|---|---|---|---|---|---|---|---|
| 0 | -2.3717 | -2.6701 | 1.9426 | 1.8547 | 0.0879 | 0.3880 | 0.0985 |
| 9 | -2.9963 | -2.9548 | 1.9306 | 1.8554 | 0.0752 | 0.3753 | 0.0617 |
| 19 | -3.0630 | -3.0537 | 1.9232 | 1.8506 | 0.0726 | 0.3726 | 0.0594 |
| 29 | -3.1425 | -3.1108 | 1.9068 | 1.8317 | 0.0751 | 0.3752 | 0.0686 |
| 39 | -3.1855 | -3.1542 | 1.8886 | 1.8097 | 0.0789 | 0.3790 | 0.0769 |
| **49 (best)** | -3.2079 | -3.1927 | 1.8882 | 1.8036 | **0.0846** | 0.3848 | 0.0892 |
| 57 (early stop) | -3.2159 | -3.1838 | 1.8844 | 1.8016 | 0.0828 | 0.3829 | 0.0846 |

Similar D_pos/D_neg gap plateau (~0.075-0.085) to the continuous run --
this internal diagnostic alone does not distinguish the two runs' quality;
`val_sub` is notably HIGHER for the discrete run throughout (0.0892 at best
checkpoint vs. continuous's 0.0391) -- expected, since discrete training
only ever sees the substitute loss at full weight (alpha=1) or zero weight
(alpha=0), never a diluted intermediate value, so its substitute-loss
average across a full epoch (mixing both alpha=0 batches, where substitute
loss is unweighted-but-still-computed-and-logged, and alpha=1 batches) is
structurally different from continuous training's smoothly-varying
per-batch alpha. This difference in the training diagnostic does NOT predict
which run does better on Recall@K -- see `ablation_results.md`, where
discrete wins decisively despite this less favorable-looking internal
number, another instance of this project's now-repeated lesson (phase 14)
that a training diagnostic's own behavior doesn't reliably predict
downstream retrieval quality.

## Endpoint / safeguard re-checks on the final trained models

- **Collapse**: not re-checked via the smoke test's `mean_pairwise_cosine`
  helper directly on the final checkpoints (that check ran pre-training on
  small subsets, see above), but the CIR eval sweep's Recall@K results being
  well above 0 and varying sensibly by alpha are themselves inconsistent
  with a collapsed embedding space (a collapsed model would produce
  near-random, alpha-invariant rankings).
- **Attention-weight-shift probe, final models**: continuous 0.188,
  discrete 0.486 (both far above the untrained baseline of 0.041) -- see
  `architecture_notes.md` and `ablation_results.md` for the full
  interpretation.
- **Endpoint tolerance check** (design decision from `results_table.md`):
  discrete checkpoint's alpha=0 Recall@K is 90.6%/95.2%/97.2% of phase 13b's
  complement-only-trained CSA-Net -- within the pre-declared +-20% band, no
  further explanation needed.

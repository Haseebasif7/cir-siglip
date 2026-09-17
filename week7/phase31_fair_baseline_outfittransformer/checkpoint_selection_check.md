# Phase 31, Step 2: Checkpoint Selection Check and Fix

## What phase 14b actually used

`week4/phase14b_outfittransformer_category_negatives/scripts/train_core.py`, lines 291-300:

```python
if val_loss < best_val_loss - 1e-4:
    best_val_loss = val_loss
    epochs_since_improve = 0
    torch.save(model.state_dict(), out_dir / checkpoint_name)
else:
    epochs_since_improve += 1
    if epochs_since_improve >= patience:
        break
```

Checkpoint selection and early stopping are both keyed to **validation loss**, never Recall@K -- Recall@K
was never computed during training at all, only once at the end via a separate script. Exactly the bug
phase 23 found and fixed for the project's own model.

## Why this matters: the loss the selector tracked was dominated by the uniformity regularizer, not ranking quality

Decomposed directly from phase 14b's own `models_random_negatives/training_curves.json`:

| Epoch | val_D_pos | val_D_neg | Margin (0.3) satisfied? |
|---|---|---|---|
| 0 | 1.2559 | 1.0785 | No (D_pos > D_neg) |
| 91 (final) | 1.4059 | 1.4035 | No (D_pos > D_neg, barely) |

**The triplet margin was never satisfied at any epoch of the entire 92-epoch run.** Yet `val_loss`
"improved" from -2.706 to -3.520 throughout, because `val_loss = triplet_loss + 1.0 * uniformity_loss`
and the uniformity term (pushing the embedding space toward broader spread) can improve independently of
whether the model actually ranks the true target above negatives. The epoch-83 checkpoint phase 14b
selected was chosen almost entirely on embedding spread, not ranking quality.

## The fix: real Recall@10 evaluation, every epoch, selection keyed to it

Added `evaluate_recall_gpu` (GPU-resident, batched, adapted from phase 14b's own
`04b_cir_eval_random_negatives.py` embed_query/embed_item_alone split) to `modal_app.py`'s `train_one`,
evaluated every epoch against **the validation benchmark** (see benchmark-discipline correction below),
selecting/early-stopping on val Recall@10 (`min_delta=0.0005`) instead of val_loss. `val_loss`/`val_D_pos`/
`val_D_neg` are still logged every epoch as secondary diagnostics -- this is what supplies the evidence
below.

## A real complication the naive fix exposed: patience=8 does not survive Recall@10's early plateau

The first attempt (A2/A3, patience=8 carried over unchanged from phase 14b's own value) early-stopped at
epoch ~9, val Recall@10 ~0.016-0.017 -- far below A1's (val_loss selection, same everything else) eventual
0.0637 at epoch 93. This is NOT evidence the selection-metric fix is wrong. Direct proof, from A1's own
curve (A1 tracks Recall@10 every epoch too, it just doesn't select on it):

| Epoch | A1 val R@10 | Note |
|---|---|---|
| 9 | 0.0169 | local high |
| 10-17 | 0.0127-0.0163 | **8 straight epochs, none beats 0.0169+0.0005** -- exactly reproduces A2's early stop |
| 18-25 | 0.0126-0.0216 | still noisy, slowly climbing |
| 26+ | 0.0200 onward | resumes a long, steady climb |
| 93 | 0.0637 | A1's own eventual best |

Recall@10 has a genuine ~17-epoch noisy plateau early in training before resuming improvement. Patience=8
was implicitly calibrated against val_loss's much smoother curve (which never triggered early stopping at
all across 100 epochs) and does not transfer to Recall@10's noisier one. Fixing this -- setting patience
large enough (25, comfortable margin above the observed 17-epoch plateau) to reflect genuine convergence
rather than transient noise -- is part of correctly implementing "select by Recall@10" (this step's own
instruction), not hyperparameter tuning (step 3, which does not touch patience as a swept variable).

## Corrected result

| id | Configuration | Patience | Val R@10 | Best epoch | Epochs run |
|---|---|---|---|---|---|
| A0 | Phase 14b's existing checkpoint, evaluated on val (no retrain) | -- | 0.0659 | -- | -- |
| A1 | Modal port, val_loss selection (fidelity check) | 8 | 0.0637 | 93 | 100 |
| A2 (broken) | + Recall@10 selection, patience unchanged | 8 | 0.0169 | 9 | 18 |
| **A2-corrected** | **+ Recall@10 selection, patience recalibrated** | **25** | **0.0786** | 84 | 100 |
| A3 (broken) | + text, patience unchanged | 8 | 0.0159 | 7 | 16 |
| **A3-corrected** | **+ text, patience recalibrated** | **25** | **0.0890** | 81 | 100 |

**A2-corrected beats A1 by +23.4% relative** (0.0786 vs. 0.0637) -- once correctly implemented (metric
swap AND patience recalibration together), the checkpoint-selection fix is real and substantial, exactly
the "first, free improvement" phase 23 found for the project's own model, though it required one extra
diagnostic step here to actually realize.

A1 fidelity check: 0.0637 (val) is in the same ballpark as A0's 0.0659 (val) and phase 14b's own reported
0.0588 (test, not directly comparable, see below) -- the Modal port reproduces phase 14b's behavior
reasonably faithfully before crediting anything to the fix.

**Provenance note**: A2-corrected and A3-corrected's reported `wall_time_sec`/epoch curves reflect only
their second Modal invocation -- the first `.map()` call was cancelled by a transient Modal error, and the
credit-safety warm-start pattern (checkpoint-persist-on-improvement, resume from last checkpoint) correctly
picked up from wherever that interrupted run left off rather than losing progress. Their "epoch 0" already
shows non-random-init Recall@10 (0.0477 / 0.0644) because of this -- the final converged numbers are a
legitimate result of continuous training, just split across two invocations. See `training_log.md`.

## Benchmark discipline correction

Phase 14b's reported `0.0588/0.1286/0.1809` was measured against
`week4/phase12_controllable_modes/data/cir_benchmark.json` -- the file every phase from 23 onward treats
as the **test** benchmark, touched exactly once. Phase 14b had no separate validation benchmark and made
no tuning decisions with it, so it never violated a rule that didn't yet exist for it -- but its number is
a **test-set measurement**, not a validation number, and must never be directly compared to any val
Recall@10 in this phase's own progression table. `week4/phase23_hyperparameter_tuning/data/cir_val_benchmark.json`
(22,595 queries, built later in phase 23) is the real validation split, and every table above uses it.

Operational rule enforced for the rest of this phase: `train_one` (`modal_app.py`) never loads
`cir_test_benchmark.json` at all -- verified directly by grep, not just by convention. The test benchmark
is opened exactly once, in the final step (`07_final_test_eval.py`).

## Verdict

This is the first, free improvement in this phase's sequence, matching phase 23's own precedent for the
project's own model -- with the added nuance that "free" here required diagnosing and fixing a second,
related problem (patience calibration) that the naive metric swap alone exposed rather than solved. Both
fixes are now baked into every subsequent step's standard configuration (`selection_metric="recall10"`,
patience raised from phase 14b's original 8 to a value informed by this diagnosis -- see `tuning_log.md`
for the exact sweep-budget patience chosen and why).

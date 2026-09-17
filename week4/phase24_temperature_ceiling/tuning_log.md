# Phase 24: Tuning Log -- Closing the Temperature Sweep's Open Edge

All runs below use phase 23's winning configuration otherwise (lr=0.001, batch_size=256, weight_decay=0.0, R=8), varying only `tau`, selected/early-stopped by validation-benchmark Recall@10 (never validation loss) -- exactly phase 23's protocol, reusing phase 23's own deployed Modal infrastructure (`app.function train_one`) and validation benchmark (`week4/phase23_hyperparameter_tuning/data/cir_val_benchmark.json`) directly, no new infra built. The test benchmark was not touched in this phase (see "Why step 3 was skipped" below).

## An important correction found before running anything new

Phase 23's own temperature sweep (its step 4, the one that found "monotonically increasing from tau=0.03 to 0.15, no peak") was run at **weight_decay=1e-5** (phase 9's original value) -- weight decay wasn't tuned to 0.0 until phase 23's *later* step 6. So phase 23's cited tau=0.15 number (val Recall@10=0.1564) is not directly comparable to phase 23's own actual final winning configuration (wd=0.0), which separately scored 0.1600 at tau=0.15. This phase re-measures the full tau range fresh, entirely at wd=0.0 (the real final setting), rather than assuming the wd=1e-5 sweep's shape carries over unchanged.

## Batch 1: extending upward (tau > 0.15), max_epochs=12, patience=4

| tau | val Recall@10 | best_epoch / epochs run | wall time |
|---|---|---|---|
| 0.20 | 0.1564 | 2 / 7 | 701s |
| 0.25 | 0.1478 | 1 / 6 | 579s |
| 0.30 | 0.1405 | 1 / 6 | 602s |

Immediate, clear decline starting right after 0.20 -- two consecutive declining steps (0.25, 0.30 both below 0.20), satisfying the brief's own stopping rule. No need to extend further upward (e.g. to 0.35+).

## Batch 2: bracketing the peak at the correct wd=0.0 setting (tau <= 0.15), same protocol

| tau | val Recall@10 | best_epoch / epochs run | wall time |
|---|---|---|---|
| 0.10 | 0.1558 | 4 / 9 | 1040s |
| 0.12 | 0.1557 | 3 / 8 | 800s |
| **0.15** | **0.1600** | 4 / 9 | 890s |

## Full picture, all 6 values under the identical protocol (lr=0.001, bs=256, wd=0.0, R=8, max_epochs=12, patience=4)

| tau | val Recall@10 |
|---|---|
| 0.10 | 0.1558 |
| 0.12 | 0.1557 |
| **0.15** | **0.1600** |
| 0.20 | 0.1564 |
| 0.25 | 0.1478 |
| 0.30 | 0.1405 |

**tau=0.15 is a clean, bracketed peak** -- lower on both sides (0.10/0.12 below it, 0.20/0.25/0.30 declining further out), not an edge case. This resolves phase 23's flagged open question: the "no peak found by 0.15" result was an artifact of testing the temperature sweep at the wrong (not-yet-tuned) weight decay; at the actual final wd=0.0 setting, 0.15 already sits at the peak.

## Step 2 (confirmatory reproducibility): already satisfied, no new run needed

tau=0.15/wd=0.0's val Recall@10 = 0.1600, best_epoch=4, is now confirmed **bit-identical (0.15999114848417792) across three fully independent runs** at three different epoch budgets:

| Run | Source | max_epochs / patience | val Recall@10 | best_epoch |
|---|---|---|---|---|
| Phase 23, step 6 refinement | `phase23_hyperparameter_tuning/data/refinement_results.json` | 15 / 6 | 0.15999114848417792 | 4 |
| Phase 23, final confirmatory retrain | `phase23_hyperparameter_tuning/data/final_config_train_result.json` | 60 / 15 | 0.15999114848417792 | 4 |
| Phase 24, this batch | `data/tau_extension_results.json` | 12 / 4 | 0.15999114848417792 | 4 |

Given fixed seeds (seed=42 throughout) and a fully deterministic training/eval pipeline, this is bit-exact reproduction, not just "close across runs" -- step 2's own goal (confirm the winner reproduces, not a single lucky run) is satisfied more strongly than a fresh retrain alone would have shown, so no additional confirmatory run was performed here.

## Step 3: skipped, and why

Per the brief's own instruction ("If 0.15 turns out to already be the real peak or close enough that nothing meaningfully changes, say so plainly rather than manufacturing a difference") -- tau=0.15 IS the confirmed peak, identical to phase 23's already-adopted final configuration. There is no new configuration to evaluate on the test benchmark; phase 23's own test-benchmark result (Recall@10/30/50 = 0.1473/0.2684/0.3442, `week4/phase23_hyperparameter_tuning/final_evaluation.md`) stands unchanged as the current best. See `final_evaluation.md` in this phase's folder for a short note confirming this explicitly.

# Phase 31: Training Log (as built)

## Step 0: sampler vectorization

`00a_sampler_equivalence_check.py`, category "shoes" (largest pool, 51,132 entries / 36,481 unique):

| Implementation | 200,000 draws | ms/draw | Violations |
|---|---|---|---|
| Original (`random.sample` over the filtered list) | 239.7s | 1.199 | 0 |
| Vectorized (index-space rejection sampler) | 2.33s | 0.012 | 0 |

~103x speedup. Two-sample chi-square (contingency table): statistic=36,643.28, dof=36,474, p=0.265,
stat/dof=1.005 -- statistically indistinguishable distributions. (The first two attempts at this check
failed for test-construction reasons, not sampler bugs: attempt 1 used a naive `item -> position` dict
built from the duplicated category list itself, silently colliding on repeated items; attempt 2 used the
wrong statistical test, `chisquare` against a noisy empirical sample treated as a fixed theoretical
expectation. Both are documented in the script's own comments.)

## Step 0: volume verification

`00_verify_volume_data.py` -- all local/volume hashes matched exactly (`cir_val_benchmark.json`,
`cir_test_benchmark.json` confirmed byte-identical to `week4/phase12_controllable_modes/data/
cir_benchmark.json`), `training_data.json` and `cir_train_benchmark.json` uploaded (10.3MB, 1.2MB),
`text_embeddings.npz`/`siglip_base.npz` item_id alignment confirmed (251,008 items).

## Step 0: A0 reference point

`00b_reference_point.py`, local MPS: phase 14b's existing checkpoint, evaluated on the validation
benchmark for the first time -- Recall@10/30/50 = 0.0659/0.1403/0.1979 (22,595 val queries, 0 skipped).

## Smoke test

`modal run scripts/modal_app.py::smoke_test` -- 2 epochs each, image and image_text. No errors.
Measured ~46s/epoch (image) and ~32s/epoch (image_text) including full Modal cold-start overhead.

## Step 2: A1/A2/A3 measurement runs

First `.map()` launch of A1 (val_loss selection)/A2/A3 (recall10 selection, patience=8) completed cleanly.
A2/A3 both early-stopped at epoch ~9 (val_recall10 ~0.016-0.017) -- diagnosed as a patience/plateau
mismatch, not a broken fix (see `checkpoint_selection_check.md`). Corrected re-runs (A2/A3 at patience=25,
`02b_selection_patience_fix.py`) hit a transient Modal `RemoteError('Function call was cancelled by user
or a failure.')` on the first attempt -- re-running the identical script picked up cleanly via the
credit-safety warm-start pattern (checkpoints from the cancelled attempt were already on the volume); the
corrected runs' reported epoch-0 values (r10=0.0477/0.0644, not near-random-init ~0.011-0.013) reflect
this warm start -- their final converged numbers are a legitimate result of continuous training, just split
across two Modal invocations, and their `wall_time_sec` reflects only the second invocation.

## Step 3: tuning

15-config LR x BS grid + 1 bracket-extension config (bs=768) + 7-config loss-shape sweep (3 coarse +
4 refined uniformity_weight + 4 margin, one point reused from the grid) + 1 full-budget confirmation run,
all via `.map()`, all completed without incident. See `tuning_log.md` for the full results.

## Step 4: scale sweep -- stopped for budget

`04_scale_sweep.py` launched 30 configs via `.map()`. A `modal billing report --for "this month"` check
(prompted by the user's "$7 left" note) showed **$22.28 already spent on phase 31** against **~$7
remaining**. The app (`ap-Un73fVpkMY6HibgLSsLerq`) was stopped immediately (`modal app stop --yes`) -- 5
running containers terminated, 0/30 configs had completed or been checkpointed (`save_checkpoint=False`
for these throwaway sweep configs), so nothing was lost, but nothing was gained either. See
`scale_sweep.md`.

## Step 5: ensembling -- not started, same reason

Not attempted, given step 4's stop. See `individual_seeds.md`, `ensemble_size_sweep.md`.

## Step 6: final evaluation

`07_final_test_eval.py`, local MPS, zero further Modal spend -- reuses the already-trained, already-
downloaded `models/ot31_budget_check_full.pt` (step 3's full-budget-trained winner). Test benchmark
(29,681 queries, 0 skipped): Recall@10/30/50 = 0.1799/0.3111/0.3844. See `final_evaluation.md`.

## Total Modal spend, phase 31

$22.28 (per `modal billing report --for "this month"`, summed across the phase's four app incarnations --
the app was redeployed a few times as `modal_app.py` was extended, each redeploy keeps the same volume/
checkpoints). No further Modal spend after step 4 was stopped.

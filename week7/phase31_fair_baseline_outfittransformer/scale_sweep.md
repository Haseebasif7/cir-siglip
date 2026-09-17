# Phase 31, Step 4: Scale Testing -- NOT COMPLETED (budget constraint)

## What happened

The scale sweep script (`scripts/04_scale_sweep.py`) was built per the plan: 5 axes (`n_heads`,
`n_layers`, `d_ffn`, `d_model`, `d_embed`), each at 2 sizes with a 3-point LR recheck, 30 configs total,
launched via `.map()` at the sweep budget (`max_epochs=60, patience=20`, matching step 3's own precedent).

It was stopped immediately after launch, before any config completed (`0/30 configs done`, nothing was
checkpointed since `save_checkpoint=False` for these throwaway sweep configs -- no partial progress lost).

## Why

This project's Modal account (`m-haseebasif5`) hit a hard budget wall mid-phase. A billing check
(`modal billing report --for "this month"`) showed **$22.28 already spent on phase 31 alone** against
**~$7 remaining** in the account when the constraint was raised. The 30-config scale sweep, at the
observed per-config cost of the tuning step (roughly $0.30-0.60 per 60-epoch T4 run), would have cost
somewhere in the $9-18 range on its own -- more than the entire remaining balance, before step 5's
ensembling (which would cost more again) or the final evaluation. Stopping immediately, rather than
letting the sweep run until credits ran out mid-batch, was the only way to guarantee a clean, honestly-
reportable stopping point.

## What this means for the phase's result

Per the brief's own "if things don't go smoothly" guidance ("if any step shows no improvement at all,
report it as a real finding, don't force an appearance of contribution where there isn't one") applied to
the inverse case here: this step was not attempted at all, and that is reported directly rather than
fabricated or silently skipped. **No claim is made about whether scale testing would have helped or hurt**
-- it is simply untested. The phase's final result (`final_evaluation.md`) is the single model from step 3
(hyperparameter-tuned, but architecturally identical to phase 14b's original transformer shape:
`d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512`), not a scale-optimized configuration.

`scripts/04_scale_sweep.py` is left in the repository, fully built and ready to run, should Modal credits
become available again -- it was verified to launch and dispatch configs correctly (the `.map()` call
itself succeeded in submitting all 30 configs to Modal before the app was stopped) before being stopped
for budget reasons, not because of any code or design problem.

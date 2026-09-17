# Phase 31, Step 5: Individual Seeds -- NOT COMPLETED (budget constraint)

Step 5 (train 10 independent seeds of the winning configuration, combine by score-averaging, identical
methodology to phases 26/28) was not attempted. This project's Modal account hit a hard budget wall mid-
phase: `modal billing report` showed $22.28 already spent on phase 31 against ~$7 remaining when the
constraint surfaced, immediately after step 4's scale sweep was stopped for the same reason (see
`scale_sweep.md`). At the observed per-run cost of a full 100-epoch training run (roughly $0.50-1.00 each
on T4), even a reduced ensemble would have meaningfully strained or exceeded the remaining balance, and
step 4's scale sweep had first claim on whatever budget-check runs were already in flight.

Seed 42 alone (step 3's fully-tuned winner, `models/ot31_budget_check_full.pt`, val Recall@10=0.1924) is
what `final_evaluation.md` reports -- a genuine single-model result, honestly labelled as such, not an
ensemble presented as one.

See `phase31_notes.md` for the full disclosure and what this means for interpreting the final numbers, and
for why the single-model result already reaching ~94-95% of phase 28's own 10-model ensemble is a striking
enough finding on its own that the missing ensemble step doesn't undermine the phase's central conclusion.

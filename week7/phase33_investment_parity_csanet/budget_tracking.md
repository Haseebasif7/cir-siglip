# Phase 33: Budget Tracking

Same methodology as phase 32's `budget_tracking.md`: `modal billing report` reports cumulative spend, not
remaining balance directly. Remaining balance is tracked as an anchored delta from the user's stated $7.00
(established at the end of phase 31, month-to-date spend $22.29818302 at that moment). Every check re-runs
the report and computes `remaining = $7.00 - (current_spend - $22.29818302)`.

**Safety margin: $1.50.** If remaining balance falls below this before launching a new paid run, stop
immediately and report whatever seeds have actually completed.

Steps 1-2 (checkpoint-selection check, text integration) ran entirely LOCALLY on the M4 Air, free -- see
`checkpoint_selection_check.md`, `text_input_integration.md`. Step 3 (tuning grid, this phase's "heavy
step" -- genuinely parallelizable, unlike the isolated local runs) ran on Modal: 15-config grid + 1 bracket
extension, total cost **$0.8987** (spend went from $23.1931 to $24.0918).

| Checkpoint | Month-to-date spend | Estimated remaining | Action |
|---|---|---|---|
| Start of step 4 | $24.0918 | $5.2064 | OK to proceed -- launch seed 42 |
| After seed 42 (cost: $0.0642) | $24.1560 | $5.1422 | OK to proceed -- launch seed 1 |
| After seed 1 (cost: $0.0652) | $24.2212 | $5.0770 | OK to proceed -- launch seed 2 |
| After seed 2 (cost: $0.0539) | $24.2751 | $5.0231 | OK, but per the brief's own step 4 target (exactly 3 seeds, not "as many as budget allows"), stop here -- 3-seed ensemble (42, 1, 2) is complete |

## Summary

| Seed | Val Recall@10 | Cost |
|---|---|---|
| 42 | 0.1075 | $0.0642 |
| 1 | 0.1008 | $0.0652 |
| 2 | 0.1028 | $0.0539 |

**Total phase 33 Modal spend: $1.0821** ($0.8987 for the 16-config tuning grid + $0.1833 for 3 seeds).
Final remaining balance: **$5.02** -- CSA-Net's small model size made every step in this phase dramatically
cheaper than phase 31/32's OutfitTransformer work (their combined spend was ~$23), leaving budget well
above the $1.50 safety margin the entire way through. Unlike both phase 31 and phase 32, this phase's
budget was never remotely close to binding.


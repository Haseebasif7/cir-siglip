# Phase 32: Budget Tracking

Balance-check methodology (`scripts/check_balance.py`): `modal billing report` reports cumulative spend
over a period, not a remaining-balance figure directly -- Modal's CLI has no direct "remaining credits"
command. Remaining balance is tracked as an anchored delta: the user stated **$7.00 remaining** at the
exact point phase 31 was stopped, which corresponds to a known month-to-date spend figure captured at that
same moment (`$22.29818302`, from `modal billing report --for "this month"`, re-verified fresh at the start
of this phase and unchanged from the phase 31 checkpoint -- confirming no new spend occurred in between).
Every check re-runs the report and computes `remaining = $7.00 - (current_spend - $22.29818302)`.

**Safety margin: $1.50** (per the brief). If remaining balance falls below this before launching a new
paid run, stop immediately and report whatever seeds have actually completed.

| Checkpoint | Month-to-date spend | Estimated remaining | Action |
|---|---|---|---|
| Anchor (start of phase 32, re-verified) | $22.2982 | $7.0000 | -- |
| Before seed 1 | $22.2982 | $7.0000 | OK to proceed -- launch seed 1 |
| After seed 1 (cost: $0.4312) | $22.7294 | $6.5688 | OK to proceed -- launch seed 2 |
| After seed 2 (cost: $0.4454) | $23.1748 | $6.1234 | OK, but per the brief's own step 2 scope (exactly seed 1 and seed 2, not "as many as budget allows"), stop here -- 3-seed ensemble (42, 1, 2) is complete |

## Summary

| Seed | Val Recall@10 | Cost | Provenance |
|---|---|---|---|
| 42 | 0.1924 | $0.00 (reused from phase 31's `ot31_budget_check_full.pt`) | phase 31's step 3 winner |
| 1 | 0.1942 | $0.4312 | trained this phase |
| 2 | 0.1929 | $0.4454 | trained this phase |

**Total phase 32 spend: $0.8766.** Final remaining balance: **$6.12** -- well above the $1.50 safety
margin the whole way through. The brief's own scope (exactly 2 additional seeds beyond the free reused
seed 42) was reached without the budget constraint actually binding this time; unlike phase 31, this phase
did not need to stop early.


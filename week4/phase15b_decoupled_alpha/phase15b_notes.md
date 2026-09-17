# Phase 15b: Fully Decoupled Alpha Conditioning -- Verdict

## The stop-condition verdict, stated plainly

**The pre-declared stop condition was NOT met.** It required both (a) the
realized Recall@10 range closing at least half the gap to phase 12c's
mechanism (~15-17% relative or more) AND (b) neither endpoint's Recall@K
dropping meaningfully below phase 15's discrete-trained endpoints. Part (a)
was cleared decisively -- the realized range (35.6% relative at Recall@10)
more than doubled the bar and even exceeded phase 12c's own reference range
(31%). **Part (b) was not cleared**: the substitute-leaning endpoint
(alpha=1) dropped to 68.4%/78.1%/81.8% of phase 15 discrete's own alpha=1
Recall@10/30/50 -- a real, consistent, one-directional regression, not
noise, exceeding this project's own ±20% "meaningful" threshold at two of
three K values.

**Per this phase's own pre-declared instruction: since the bar was not
cleared, no further variant is attempted (no alpha-dependent loss weight, no
different conditioning encoding, nothing else). Phase 15's discrete-trained
checkpoint remains the final, adopted result for this architectural
thread.** This phase's decoupled checkpoint is NOT adopted as the new best
configuration, despite comfortably beating phase 15 discrete on the range
metric and on both diagnostic-strength probes (attention-weight shift,
smoothness gap) -- the compound bar required BOTH range AND endpoint
preservation, and trading substitute-side quality for a wider range was
exactly the failure mode the endpoint-regression clause was written to
catch.

## What actually happened, mechanistically (useful even though the bar wasn't cleared)

Removing alpha from the loss envelope worked exactly as diagnosed in phase
15: the attention-weight-shift probe now measures 1.9955 (vs. phase 15
discrete's 0.486), essentially the theoretical maximum of 2.0, uniformly
across all 121 category pairs (min 1.9545, max 1.9998 -- the tightest
spread this project has measured). The smoothness gap (0.6958) is the
strongest of any conditioning mechanism evaluated in this project,
including phase 12c/12d's own simple mechanism (0.475). **The double-duty
confound diagnosed in phase 15 was fully resolved architecturally** -- this
part of the phase 15b hypothesis was correct and is now confirmed, not just
plausible.

But full resolution of the confound did not translate into a symmetric,
smooth, usable dial. The realized shape (`results_table.md`) is a
near-flat-then-sharp-drop transition, not a linear interpolation: Recall@K
and every diagnostic stay essentially constant (even slightly RISING) from
alpha=0.0 to ~0.5-0.6, then fall sharply toward alpha=1.0. The complement
pathway (alpha near 0) improved slightly over phase 15 discrete's own
complement endpoint, essentially matching phase 13b's fixed CSA-Net at
R@50. But the substitute pathway (alpha near 1) got WORSE than every prior
version of this architecture, including phase 15's own continuous-trained
checkpoint. A specific, concrete oddity: the overlap-with-raw-SigLIP
diagnostic actually DECREASES from alpha=0 to alpha=1 late in the sweep
(0.099 -> peaks at 0.105 around alpha=0.6 -> drops to 0.075 by alpha=1.0) --
the opposite direction from every other mechanism in this project, where
higher alpha (more substitute-leaning) means MORE similarity to raw SigLIP,
not less.

**A plausible explanation, not tested further here (out of scope per this
phase's own "stop, don't iterate" instruction)**: with both objectives now
pulling on the shared parameters at full, undiluted strength every single
step (rather than phase 15's alternating/enveloped exposure), the shared
`proj`/`masks` parameters may have been pulled toward a solution that favors
the complement objective's own structure (CSA-Net's native mechanism, which
this architecture is built on) at the expense of the substitute objective's
needs -- i.e. training pressure was not just decoupled from alpha, it was
also now perfectly SIMULTANEOUS and un-annealed (no LR-style softening of
which objective dominates at a given point in training), which may have let
the numerically larger-scale complement pathway's gradient shape the shared
representation more than the (still `weight_sub`-scaled, but structurally
different) substitute pathway could counteract. This is speculation, offered
as a plausible mechanism for a future investigation, not a proven
explanation -- per this phase's own scope, no further variant was attempted
to test it.

## What this does and doesn't establish

- **Does establish**: the alpha double-duty confound identified in phase 15
  is real and fully fixable architecturally (the attention-weight-shift
  probe and smoothness gap prove this decisively) -- but fixing it alone is
  NOT sufficient to produce a good controllable mechanism. A wide, smooth
  dial and balanced endpoint quality are separate achievements, and this
  phase's one bounded attempt got the former without the latter.
- **Does establish**: this project's own pre-declared, compound stop
  condition (range AND no endpoint regression) is a meaningfully different,
  stricter bar than range alone -- had the bar only checked range, this
  phase would have been reported as a clean, decisive win. The
  endpoint-regression clause caught a real quality trade-off that a
  range-only metric would have missed entirely, validating the choice to
  write it into the stop condition in advance.
- **Doesn't establish** why the substitute endpoint specifically regressed
  (the shared-parameter-competition explanation above is plausible, not
  proven) -- not investigated further, per this phase's own scope.
- **Doesn't establish** that no variant of decoupled training could ever
  clear the full bar (e.g. a per-objective learning-rate schedule, or
  annealing the objectives in rather than applying both at full strength
  from step 0) -- genuinely untested, explicitly out of scope for this one
  bounded attempt.

## Where this leaves the architectural thread

Per this phase's own instruction, **phase 15's discrete-trained checkpoint
stands as the final result for the CSA-Net-plus-controllable-dial
architectural thread.** No further iteration on this specific mechanism is
implied by this phase's outcome -- the thread is closed, not paused pending
more tuning. Phase 12c/12d's simpler additive mechanism remains this
project's best-validated controllable mechanism overall (best combination of
Recall@K, range, and smoothness, with no endpoint trade-off). The next step
for this project, per the brief's own "do not do yet" instruction, is a
paper writeup -- not a further variant of this thread.

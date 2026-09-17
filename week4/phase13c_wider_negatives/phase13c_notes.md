# Phase 13c: Widening the Negative Pool -- Verdict

## The explicit stop condition, as decided in advance (step 3 of the brief)

> Compare the final D_pos/D_neg gap against phase 13b's 0.082. If the gap
> closes by at least half the remaining distance toward zero (drops to
> roughly 0.041 or lower), that counts as a real, meaningful improvement...
> If it doesn't close by at least that much, or the sign still never flips,
> stop here.

**Result: the condition is not met, and not by a wide margin.** The final
gap is **0.120** (vs. phase 13b's 0.082) -- it needed to drop to ≤0.041 and
instead it is *larger* than the baseline it was supposed to improve on, at
every epoch of training, not just at the end (see `training_log.md`'s
epoch-by-epoch table). The sign never flipped either (D_pos stayed above
D_neg throughout, same as phase 13b).

## What this means

**Widening the negative pool from 10 to 20 mined candidates did not close
CSA-Net-on-SigLIP's ranking-diagnostic plateau -- it made it worse.** This
is a real, useful negative result, not a null one: it rules out "the model
just needed harder/more numerous negatives from what was already mined" as
the explanation for phase 13b's plateau. Since the SAME mined pool was used
(just more of it per sample), and the same architecture and loss, the most
plausible reading is that the extra negatives diluted the training signal
per step rather than sharpening it -- with `min` aggregation, adding more
already-similar (same-category, SigLIP-nearest-neighbor) candidates doesn't
necessarily surface a harder negative than the 10 already used; it may just
add more low-information terms to the batch that the min-aggregated loss
ends up not learning as efficiently from, given the same effective step
budget (40 epochs either way).

**Per this phase's own explicit instruction, no third variant is being
tried** (not a larger attention sub-network, not a different aggregation
function). The comparison between CSA-Net's mechanism and this project's
own phase 12c mechanism stands exactly as phase 13b already, honestly,
reported it: mixed, not a clean win either direction. CSA-Net's mechanism
(evaluated with 10 negatives, phase 13b's checkpoint -- the better of the
two ranking-diagnostic results) beats phase 12c's substitute mode, loses to
blend and complement modes, and all three lose to phase 9's plain
compatibility model. **Phase 13b's `results_table.md` and
`phase13b_notes.md` remain the reference point for this comparison; this
phase does not supersede or improve on them.**

## Required-output checklist

- `training_log.md` -- done: the training run (including the honest
  laptop-sleep interruption and clean restart), the full epoch-by-epoch
  D_pos/D_neg comparison against phase 13b.
- `results_table.md` -- **not produced**, per the brief's own step 4 ("only
  if step 3's condition is met") -- it was not met.
- This file -- verdict: **stop condition not met; the gap got worse, not
  better; phase 13b's result stands as final for this baseline.**

## Do not do yet (per the brief)

No third variant of this fix (larger attention sub-network, different
aggregation function) was attempted, as instructed. The OutfitTransformer
work is explicitly separate, next, using different (new) budget -- not
started here.

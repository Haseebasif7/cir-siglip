# Phase 16b: Tail-Restricted Training -- Stopped at Step 1, Data-Insufficient

## What this phase attempted

Phase 16 built a working but structurally weak controllable relevance/
tail-exposure dial, diagnosed precisely: both modes trained to predict the
same `also_buy` edges, differing only by IPS reweighting, a gentler lever
than genuinely different prediction targets. This phase's one bounded
attempt to fix that (mirroring phase 8's fix to phase 7's own weak
compatibility model: restrict training to a genuinely different edge
subset, not reweight the same one) was to restrict tail-exposure mode's
training edges to those where the target item is tail-tier, no IPS
weighting this time, single-variable test.

## What actually happened

Step 1 (`edge_filtering_summary.md`) found the tail-tier-restricted edge
set is **125 edges total (117 train / 8 val), across only 37 unique
tail-tier target items** -- out of phase 7's full 76,293-edge set. This
falls dramatically short of any reasonable "meaningful to train on" bar
(this phase's own pre-declared threshold was 1,000 train edges; 117 misses
it by an order of magnitude, and with only 37 unique targets the model
would have essentially nothing to generalize across even if it trained).
Batch size in phase 16's own recipe is 128 -- 117 train edges can't even
form one full training batch.

**Per the brief's own explicit instruction ("if the tail-tier-restricted
edge count from step 1 is too small to train on meaningfully, stop and
report that directly, don't proceed on a thin signal"), this phase stops
here.** Steps 2-4 (training, stop-condition evaluation, and the full
alpha-sweep/field-metric/smoothness evaluation) were never reached --
there is no model checkpoint, no `results_table.md`, no
`smoothness_check.md` with real numbers for this phase. Stub files exist
at those paths pointing back to this file, so the required-output
checklist has a clear trace rather than a silent gap.

## Why the count is this small -- a real, structural finding, not a bug

The tier breakdown of ALL 76,293 edge targets (`edge_filtering_summary.md`):
**97.78% head-tier, 2.05% mid-tier, only 0.16% tail-tier.** This is not a
join bug or a sampling fluke specific to this phase -- it's a direct,
mechanistic consequence of what an also_buy edge TARGET *is*: an item
becomes a co-purchase target because other people's purchases pointed at
it, and phase 3's `ref_count`/tier definitions are built from exactly that
same reference-counting logic. High-ref_count (head-tier) items are, by
construction, the ones that get pointed at most often -- so they dominate
the target side of any also_buy edge list almost completely, and genuinely
tail-tier items essentially never appear as a target at all. This is a
sharper, more extreme version of the same skew phase 3 already documented
in phase 1b's eval sample (96.2% head-tier there, from BFS sampling
favoring well-connected items) -- here the skew isn't from how the sample
was built, it's inherent to what a co-purchase edge target structurally is.

## Verdict

**Phase 16's result stands as this thread's honest, final answer for this
specific mechanism design, for now.** The tail-restriction fix that worked
for phase 8's plain compatibility model doesn't have enough raw data to
even attempt here, because the tail-tier population essentially doesn't
exist on the target side of this project's also_buy edge graph. Per the
brief's explicit instruction, no third variant is proposed, and the stop
condition (Hit-Rate axis clearing a real gap, smoothness gap at least
doubling phase 16's 0.0534) was never evaluated, because there was nothing
to train and evaluate. Reopening this specific fix would require a
different data source for tail-exposure positives entirely (e.g. widening
what counts as a valid tail-exposure "positive" beyond also_buy edges
specifically), not a fresh attempt at this same restriction with the same
data -- explicitly flagged as a possible future direction, not attempted
here, per this phase's own bounded scope.

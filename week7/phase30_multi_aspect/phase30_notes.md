# Phase 30: Honest Interpretation

## Headline: single-seed gate says no, close to baseline, not a collapse, and the phase stops here

Per the brief's own explicit "stop early if no signal" instruction (step 3), a single seed is trained
and checked against phase 27/28's own single-seed text-only result before any ten-seed budget is spent.

| Configuration | Val Recall@10 |
|---|---|
| Phase 27/28 text-only, single seed (42), mean-pool | 0.1819 |
| Phase 30 multi-aspect, single seed (42) | 0.1781 |

Relative gain: **-2.09%**, just past the "same or worse, stop" line the brief draws (the required bar to
proceed was +2%). Per the brief's own instruction, this phase does not train the remaining nine seeds,
does not run an ensemble size sweep, and does not touch the test benchmark. `individual_seeds.md`,
`ensemble_size_sweep.md`, and `final_evaluation.md` are intentionally not produced -- they're conditional
on reaching full-ensemble scale in the brief's own required-outputs list, and this phase never gets there.

Worth stating plainly up front: unlike phase 29's cross-attention result (-32.5%, a clear, dramatic
regression), this result lands almost exactly at the baseline. The architecture is not badly broken --
it's a small, real cost for meaningfully more complexity, which is itself the finding worth reporting
honestly rather than rounding up to "basically a tie" or down to "a failure."

## Does multi-aspect representation improve on plain projection? No, not at the single-seed level, and the phase never reaches ensemble scale to test the harder version of that question

The single-seed result answers the question the gate exists to ask cheaply: no, this specific
architecture does not clear the bar needed to justify testing it at the full ten-seed scale phase 28 used
to confirm text's own single-seed gain wasn't noise. Whether multi-aspect representation could help at
ensemble scale despite a single-seed loss is left genuinely untested -- the brief's own gate discipline
exists precisely so that question isn't chased with a full budget every time a single seed comes back flat.

## What do the learned aggregation weights actually look like: uniform, one-hot, or genuinely varied?

Genuinely varied, not collapsed to either failure mode the brief warned about. `aggregation_qualitative.md`
samples 8 validation queries: aggregation weights range from 0.041 to 0.567 across individual
(query, aspect) pairs, and the "general" aspect's weight has the widest spread across queries (std 0.173,
vs. 0.056-0.098 for the other three) -- the mechanism is clearly query-sensitive, not stuck outputting a
fixed distribution regardless of input. At the same time, the sample's mean normalized entropy is 0.912,
fairly close to the uniform ceiling of 1.0 -- variation exists but the mechanism is still, on average,
closer to soft-averaging-with-a-lean than to sharply committing to one or two aspects per query. Both
things are true simultaneously and both are reported here rather than picking whichever framing sounds
better.

The aspect-head collapse check (`aggregation_qualitative.md`'s pairwise cosine table, 2000-item sample)
rules out the other obvious failure mode directly: all six pairwise cosines between aspect heads' outputs
are near zero (max |cosine| = 0.055, `general` vs. `visual`), meaning the four heads learned genuinely
distinct, non-redundant functions of the shared trunk, not four copies of the same thing wearing
different names.

## What do the per-aspect similarity contributions tell us, and what's the most likely reason this didn't help?

This is where the qualitative check earns its keep beyond confirming the mechanism "works" mechanically.
Look at query 212856802 (`aggregation_qualitative.md`): the true target scores 0.300, but "random
negative 1" scores 0.587 -- the model ranks a random negative ABOVE the true target here. Decomposing the
score: negative 1 beats the true target on all three of visual (0.491 vs -0.065), form (0.633 vs 0.346),
and semantic (0.475 vs 0.340), and ties it on general (0.818 vs 0.817). The aggregation weights
themselves are reasonably balanced for this query (0.294/0.327/0.217/0.161) -- this isn't a case of one
aspect's weight being pathologically wrong. The failure is upstream of aggregation: the 32-d aspect
representations themselves aren't separating this particular negative from the true target in most of
the four subspaces.

The most likely, mechanistic reason, consistent with the training curve's own pattern (see
`training_log.md`): splitting a 128-d representation into four independent 32-d subspaces costs real
discriminative capacity per subspace. A 32-d cosine similarity is a geometrically noisier, less separable
signal than a 128-d one -- in lower-dimensional spaces, the expected similarity between two "somewhat
related but not matching" unit vectors concentrates less tightly around zero, so near-miss negatives
score less distinguishably from true positives than they would in the full 128-d space phase 27/28's
single head operates in. The training curve shows this directly: the best epoch is the one where the
aggregation mechanism is closest to uniform (entropy 0.92, effectively spreading the score across all
four noisier 32-d subspaces roughly equally, which is closer to what a single, larger, less-decomposed
representation would do), and quality degrades monotonically as training pushes the aggregation weights
to specialize further away from uniform (entropy falling to 0.63 by epoch 4) -- the model leaning harder
into its lower-capacity per-aspect subspaces makes things worse, not better.

## Is this "aspect decomposition is the wrong inductive bias" or "K=4 didn't capture the right aspects"? Neither cleanly -- it's a capacity cost from decomposition itself, not a failure of the heads to differentiate

This result does not read as the aspect heads failing to learn anything ("uniform collapse" -- ruled out,
weights are query-varying and heads are non-collapsed/near-orthogonal) or as a plain implementation bug
(the scoring formula was verified locally via a gradient sanity check before any Modal run, and matches
the brief's exact specification; see `architecture_notes.md`). It also isn't clear evidence that
complementary matching has no genuine per-aspect structure to exploit -- that hypothesis was never
directly falsified here. What the evidence supports is narrower and more specific: at K=4 aspects of
32-d each, splitting the representation costs more in per-subspace discriminative capacity than it gains
from letting the model weight aspects differently per query, on this particular dataset and benchmark.
Whether a different split (fewer, larger aspects; or the same K=4 but with more total output dimensions
than phase 27/28's 128-d budget) would change this balance is untested and, per the brief's own "don't
exceed K=4 in this first version" and "don't over-engineer the first test" instructions, not attempted
here.

## Is the gain large enough to justify the complexity? Not applicable -- there is no gain

The brief's final honest-interpretation question doesn't apply in the usual direction: multi-aspect
representation did not improve on plain projection at the single-seed level, so there's no complexity
tradeoff to weigh. If anything, the added complexity (516 extra parameters, four separate normalized
subspaces, a learned query-side aggregation layer) bought a small quality cost instead of a gain.

## Final, honest verdict

**Phase 28's mean-pooled text-only ensemble (Recall@10/30/50 = 0.1904/0.3267/0.4079) remains this
project's best result.** Multi-aspect item representation with learned per-query aggregation, as
implemented and trained in this phase, does not beat single-vector mean-pooled projection at the
single-seed level (0.1781 vs. 0.1819, -2.1% relative) and does not pass the brief's own gate to proceed
to ensemble scale. The architecture works as designed -- gradients flow correctly, the four aspect heads
learn genuinely distinct (near-orthogonal) representations, and the aggregation layer produces real,
query-sensitive, non-collapsed weights -- but the decomposition itself appears to cost more in
per-subspace discriminative capacity than the query-conditioned reweighting recovers, evidenced both by
the training curve (quality degrades as the mechanism specializes away from uniform) and by direct
inspection of a concrete ranking failure (a near-miss negative that resembles the query context across
multiple 32-d subspaces simultaneously). This is the "well-motivated architectural change that didn't
help" outcome named in the brief as a valid finding in its own right, with a specific, evidence-backed
mechanism (representational capacity lost to decomposition) rather than a vague "it just didn't work."

As stated in the brief's framing reminder, and consistent with the fairness-gap framing already applied
to the report: this result is a matched-conditions comparison against phase 28's own text-only single
seed on this project's own CIR benchmark reproduction, not a claim about OutfitTransformer's published
numbers.

## Reference configuration, for the record

- **Architecture**: shared `Linear(1536,1024)` trunk (identical to phase 27/28's own hidden layer)
  followed by four independent `Linear(1024,32)` aspect heads (`visual`, `form`, `semantic`, `general`),
  each independently L2-normalized, plus a `Linear(128,4)` softmax aggregation layer over the
  concatenated query aspect vectors. 1,705,604 total parameters, 516 more than phase 27/28's single
  `Linear(1024,128)` head -- see `architecture_notes.md`.
- **Training data/hyperparameters**: identical to phase 27/28's text_only configuration (same base_repr
  construction, same positive edges, same negative sampling, same lr/batch size/temperature/weight
  decay), per the brief's explicit instruction to hold everything fixed except the projection.
- **Result**: single-seed (42) val Recall@10 = 0.1781, best at epoch 0, vs. phase 27/28's 0.1819. Gate:
  NO-GO. No ensemble trained, no test-benchmark evaluation performed.
- **Checkpoint**: `week7/phase30_multi_aspect/models/multiaspect_seed42.pt` -- kept for
  reference/reproducibility, not a recommended production configuration.

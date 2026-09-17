# Phase 16c: Attribute-Based Tail-Exposure Signal -- Notes

## The full three-phase ablation, told as one story

This is the honest, complete account of three attempts at giving
tail-exposure mode a training signal that actually produces a strong
controllable dial. It should be read and eventually written up as one
deliberate ablation, not three disconnected tries.

**Phase 16: reweight the same behavioral signal.** Trained tail-exposure
mode on the full `also_buy` edge set, reweighted by inverse propensity
scoring. Result: real but weak, correctly-directioned movement everywhere
(Hit-Rate axis nearly flat, ref_count/APRI moved the right way but
modestly, smoothness gap 0.0534, ~9x smaller than phase 12c/12d's
reference mechanism). Diagnosis, confirmed directly via mode-vector
inspection: both modes predicted the same underlying structure, only
reweighted, and the two learned mode vectors stayed close together
(cosine similarity 0.55, small norms relative to the shared base
projection) -- too gentle a lever to push the modes apart.

**Phase 16b: restrict to a genuinely different edge subset.** Mirrored
phase 8's own successful fix to phase 7's weak compatibility model:
restrict training to `also_buy` edges where the target is tail-tier,
instead of reweighting. Result: stopped at the data check. Only 125 such
edges exist out of 76,293 (117 train, 37 unique targets) -- a genuine
structural property of behavioral co-purchase data, not a bug: 97.78% of
all also_buy edge TARGETS are head-tier, because an item becomes a
co-purchase target precisely by being pointed at often, which is what
head-tier means by construction. Popular items dominate the target side of
any organic co-purchase log; there is no way to restrict to tail-tier
targets and still have enough data, with this signal.

**Phase 16c (this phase): use a data source with no popularity-driven
scarcity.** Product attribute data (fine-grained category, phase 8's
already-built `product_types.json`) exists uniformly for every item
regardless of purchase popularity -- sidesteps phase 16b's wall directly.
Built 37,922 directed attribute-based positive pairs (214 category groups,
deliberately stratified toward tail-tail and tail-other pairs), achieving
**50.9% tail-tier edge targets** (vs. phase 16b's 0.16%) -- confirmed this
is a genuinely different signal, not a disguised repeat (1.74% directed
overlap with the also_buy edges phases 16/16b used). Brand was attempted
as a second attribute but dropped after the source throttled to an
impractical ~2.5-3hr projected fetch time (`logs/brand_extraction_report.md`);
category alone proved sufficient given the pool's healthy 45.3%-tail
item-level composition.

## What actually happened when trained and evaluated

Training converged cleanly (50 epochs, early-stopped, no collapse
throughout -- `loss_balancing_check.md` shows a well-balanced calibration,
gradient-norm ratio 1.16x, comfortably inside the [0.1, 10] band).

**The mechanistic fix worked exactly as diagnosed -- almost too well.**
The mode-vector diagnostic (`logs/mode_vector_diagnostics.md`) confirms
the two modes genuinely diverged this time: cosine similarity between
`mode_relevance` and `mode_tail` dropped from phase 16's 0.5525 to
**0.0590** (near-orthogonal), with both vectors also growing modestly in
magnitude. The smoothness check confirms this translated into real
behavioral movement: adjacent-vs-distant overlap gap is **0.1261, 2.36x
phase 16's 0.0534**, clearing this phase's own pre-declared bar (at least
double) decisively, and the decay is still monotonic.

**But the direction of that movement is wrong, not just weak.** Both axis
checks fail, and not narrowly:

- **Axis 1 (relevance should clearly beat tail on Hit Rate): FAILS, and
  REVERSED.** Hit Rate@5 goes from 0.2810 (tail, alpha=0) down to 0.2783
  (relevance, alpha=1) -- relevance mode retrieves slightly WORSE Hit Rate
  than tail mode, the opposite of the intended relationship. Same pattern
  at K=10 (0.3563 -> 0.3510).
- **Axis 2 (tail should retrieve lower ref_count than relevance): FAILS,
  and REVERSED.** Mean ref_count@5 goes from 17.97 (tail) down to 17.59
  (relevance) -- relevance mode retrieves LESS popular items on average,
  backwards from what the mode names promise.
- **Field-standard metrics tell the same reversed story**: RPI decreases
  monotonically from 0.5072 (tail) to 0.4972 (relevance) at N=10 -- head-tier
  fraction goes DOWN as alpha moves toward "relevance." Coverage@N and
  APRI show the same small, reversed trend. Full numbers: `results_table.md`.

**This phase's own pre-declared success bar (Hit Rate axis clears clearly
AND smoothness gap doubles) is NOT met.** The smoothness half is cleared
decisively; the Hit-Rate half doesn't just fail to clear, it fails in the
wrong direction. Per the brief, this is reported plainly as the third data
point in the ablation, and no fourth variant is proposed automatically.

## Why the direction reversed -- two checked explanations, not guesses

**First hypothesis tested and NOT confirmed**: that tail-exposure mode
learned a same-fine-grained-category-clustering axis (an artifact of
training on same-category pairs) rather than a tail-popularity axis, and
that this happened to correlate with the also_buy Hit-Rate ground truth
better than relevance mode did (since phase 6/8 already found ~74% of
also_buy edges are same-category). Checked directly
(`logs/category_clustering_diagnostic.md`): the fraction of top-5
retrievals sharing the query's fine-grained category is **essentially flat
across the whole alpha sweep (0.4405-0.4419)** -- tail mode does NOT
retrieve more same-category items than relevance mode does. This
hypothesis does not hold up and is reported as a checked, ruled-out
explanation, not silently dropped.

**Second explanation, directly evidenced: capacity competition degraded
relevance mode's own quality.** Relevance mode's best validation loss got
measurably WORSE in this phase's joint training than in phase 16's:
**0.9735 (phase 16) vs. 1.1272 (phase 16c), a 15.8% relative increase**,
despite relevance mode's own training recipe (full also_buy edges,
unweighted MNRL loss) being completely unchanged between the two phases.
The only thing that changed is what the SHARED trunk had to simultaneously
satisfy -- phase 16's gentler, IPS-reweighted-but-same-population tail
signal, vs. phase 16c's genuinely different, now near-orthogonal
attribute-based signal. A much more forceful, structurally different
co-training partner pulled the shared parameters further from what
relevance mode alone would have learned, degrading relevance mode's own
retrieval quality as a side effect. This is the same capacity-competition
pattern this project already documented in phase 12c (complement mode's
validation loss and Recall@10 degrading as the substitute objective's
pull on shared parameters grew, phases 12->12b->12c) -- direct precedent
for exactly this kind of tradeoff, now observed in a different mechanism
design.

**Combined verdict**: the fix accomplished exactly what it was designed to
do at the representational level (push the two modes into genuinely
different, now near-orthogonal subspaces, producing more than double the
retrieval movement) -- but it did so at a real cost to relevance mode's own
quality, strong enough to flip the sign of both axis checks rather than
just fail to clear them. Stronger differentiation is not automatically
better differentiation if the shared trunk can't accommodate both
objectives without one degrading the other.

## Verdict for this three-phase thread

Per the brief's own instruction, **no fourth variant is proposed
automatically**. The full 16/16b/16c sequence is the honest, complete
account of what was tried for this specific mechanism design (a single
shared projection with two additive mode vectors, trained jointly on two
signals for the same item population): naive reweighting under-separates
the modes (phase 16); the obvious data restriction is structurally
impossible for behavioral data (phase 16b); a properly chosen alternative
data source over-separates the modes relative to what the shared trunk can
absorb without degrading the other objective (phase 16c). That progression
-- gentle lever too weak, natural restriction structurally blocked, a
stronger lever strong enough to reverse the intended relationship via
capacity competition -- is itself a genuine, citable methodological finding
about designing controllable dual-objective retrieval mechanisms on a
single shared projection: the mode-vector-divergence fix and the
shared-capacity cost are two sides of the same coin, and this ablation is
the first place in this project's history where that tradeoff was measured
directly rather than assumed. If this thread is reopened, the natural next
lever (not attempted here, out of this phase's bounded scope) is not a
fourth data source but an architectural one: giving each mode some
dedicated (non-shared) capacity, so a stronger differentiating signal
doesn't have to come at the shared trunk's expense.

## Documentation

`PROJECT_MEMORY.md` updated to frame phases 16, 16b, and 16c as one
coherent investigation.

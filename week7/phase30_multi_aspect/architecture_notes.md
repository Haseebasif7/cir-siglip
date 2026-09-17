# Phase 30: Architecture Notes

## What changes from phase 27/28, and what doesn't

Everything upstream of the projection head is identical to phase 27/28's text_only
(`use_category="none"`) configuration: `base_repr(i) = normalize(concat(image(i),
text(i)))`, 1536-d, computed once for the whole catalog from `siglip_base.npz` +
`text_embeddings.npz` (reused as-is from `week7/phase27_text_and_category/data/`,
no new extraction needed). Training edges, negative sampling, in-batch
false-negative masking, and the MNRL contrastive framing are all unchanged from
phase 9/23/25/26/27/28.

The only variable under test: what the projection head does with that 1536-d
vector. Phase 27/28 pass it through `ProjectionHeadGeneral` (`Linear(1536,1024)
-> ReLU -> Dropout -> Linear(1024,128)`, L2-normalized) to get one 128-d vector
per item. Phase 30 replaces the final `Linear(1024,128)` with four independent
`Linear(1024,32)` heads, each producing its own L2-normalized 32-d aspect
vector, plus a small learned aggregation layer that combines four per-aspect
similarity scores into one final score, per the brief's exact formula. See
`scripts/model.py`.

## Parameter count

Directly measured (`scripts/model.py`, `MultiAspectProjectionHead`):

| Component | Params |
|---|---|
| Shared trunk (`Linear(1536,1024)`) | 1,573,888 |
| Four aspect heads (`Linear(1024,32)` x4) | 131,200 |
| Aggregation layer (`Linear(128,4)`) | 516 |
| **Total** | **1,705,604** |

Phase 27/28's `ProjectionHeadGeneral` (`Linear(1536,1024)` + `Linear(1024,128)`)
totals 1,705,088 params. The difference is exactly 516 -- the aggregation
layer. Splitting the 128-d output into four independently-normalized 32-d
heads costs nothing in raw parameter count versus one 1024->128 head (both are
1024 x 128 = 131,072 weights + 128 biases, just partitioned differently); the
entire architectural change adds only the 516-parameter aggregation layer.
This is worth stating plainly: if this phase shows a real gain, the gain
cannot be attributed to added capacity, since capacity is essentially
unchanged. Any gain would have to come from the inductive bias itself
(per-aspect L2-normalization forcing four independent unit-norm subspaces,
plus query-conditioned reweighting across them) rather than from more
parameters to fit with.

## Query-side aggregation, not candidate-conditioned

Per the brief, the aggregation weights are computed **from the query only**
(the concatenated four aspect vectors of the mean-pooled context), never from
the candidate being scored. This is a deliberate and important difference
from phase 29's candidate-conditioned cross-attention: phase 29's dead-gradient
problem and its diagnosed train/eval mismatch both stemmed specifically from
attention being conditioned on which candidate is being scored, which only
ever sees the true positive during training. Nothing here is candidate-side.

Because the aggregation weights don't depend on the candidate, the training
procedure and the evaluation procedure use the exact same mechanism, not an
approximation of it: at training time the anchor (context length 1) computes
its own aspect vectors and its own aggregation weights, exactly the same
computation a genuine multi-item context performs at evaluation time (mean-pool
each aspect across context items, renormalize, then run the aggregation
layer). There is no analogue of phase 29's "trained on the true positive only"
approximation to flag here -- the query-side design sidesteps that problem
structurally, not by a stated simplification.

## Training-time anchor-as-query-of-size-1, per phase 27/28's own precedent

Identical convention to phase 27/28: during training the anchor plays the
query role with context size exactly 1 (mean-pooling over one item is that
item). This was already established as sound in phase 27/28 for the
single-vector case; it applies here per-aspect without any new assumption,
since each aspect's mean-pool-then-renormalize step is linear in the same way
the single-vector case was.

## Scoring formula (exact, from the brief)

For query aspects `q_k` (k=1..4, each a unit 32-d vector, mean-pooled from
context items' aspect_k vectors and renormalized) and a candidate's aspect
vectors `c_k`:

```
similarity_k = q_k . c_k                      (per-aspect cosine similarity)
weights = softmax(agg_layer(concat(q_1..q_4))) (query-only, 4 numbers summing to 1)
score = sum_k weights_k * similarity_k
```

Implemented via `einsum` (`inbatch_scores`, `aspect_score` in `model.py`) for
batched in-batch positives / negatives during training, and for full candidate
pools during evaluation -- confirmed mathematically identical between the two
paths (both reduce to the same per-(query,candidate) sum over k).

## Aspect head naming is documentation of intent, not an enforced constraint

The four heads are named `aspect_head_visual`, `_form`, `_semantic`, `_general`
per the brief's stated targets, but nothing in the loss ties any head to any
specific semantic content -- the names describe the intended role, not a
guarantee. Step 6's qualitative check is what actually determines whether the
heads learned anything resembling this intended split; `phase30_notes.md`
reports that honestly regardless of what the names suggest.

## Sanity check performed before any Modal run

A local forward/backward pass on random data (`scripts/model.py`'s functions,
batch=8) confirmed: loss computes without error, gradients are nonzero into
every component (trunk, all four aspect heads, aggregation layer), and
aggregation weights sum to 1 per row as expected from softmax. At random
init, mean aggregation-weight entropy (normalized) was 0.998 -- expected,
since an untrained aggregation layer with small random weights outputs
near-uniform softmax.

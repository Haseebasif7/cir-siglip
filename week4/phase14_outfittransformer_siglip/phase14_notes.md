# Phase 14: OutfitTransformer's Set-Encoder Mechanism -- Honest Interpretation

## Headline result

This configuration reaches only **21.0% / 32.7% / 41.4%** of
OutfitTransformer's published Recall@10/30/50, and scores **below every
other configuration evaluated on this project's CIR harness so far,
including raw untrained SigLIP** (see `results_table.md`). This is a
genuinely surprising result given how well the model's own training
diagnostic converged (D_pos/D_neg gap closed to ~0.006, the tightest of
any mechanism this project has trained -- see `training_log.md`). Both
findings were checked directly rather than assumed; neither is a bug or an
embedding-collapse artifact (confirmed: final-model candidate embeddings
have mean pairwise cosine similarity 0.004, i.e. well spread out, and the
evaluation harness itself is the exact same code/benchmark phase 13b
already validated).

## Why the training diagnostic converged well but Recall@K did not: a plausible, partially quantified mechanism

The repo's own training objective (`InBatchTripletMarginLoss`, confirmed in
`architecture_notes.md`) treats every OTHER outfit's target item in the same
batch as a negative, with **no category restriction**. The CIR evaluation
task, by contrast, restricts every query's candidate pool to the target's
own category (thousands of same-category candidates) -- confirmed
consistent with phases 9/12/13's own established protocol. These are
different tasks in a way that matters:

- Directly measured on 200 random 96-outfit training batches: only
  **13.1%** of ALL possible in-batch negative pairs share the same category
  as each other (roughly what you'd expect from 11 categories of uneven
  size) -- meaning the large majority of in-batch negatives are trivially
  distinguishable from the true target by category alone, something the
  eval task never allows (every eval candidate already IS the target's own
  category).
- The SPECIFIC negative the triplet loss actually attends to (the hardest
  one per anchor) is less trivial than that raw population suggests --
  measured via a static raw-SigLIP-distance proxy, **77.2%** of the time the
  closest in-batch item to a given target is already the SAME category
  (visual similarity is itself category-correlated, so the "hardest"
  negative naturally skews same-category even with no explicit filtering).
  But this is a proxy on the ORIGINAL SigLIP space, not a guarantee about
  what the trained model's own reshaped embedding space does 90 epochs in,
  and it's still only ONE hardest negative per anchor, not a full
  same-category candidate pool of thousands.
- The loss is also satisfied **in aggregate mean** (`val_D_pos` and
  `val_D_neg` are averages over the whole validation set) -- a batch-mean
  triplet loss can be driven very low by many different embedding
  configurations, not all of which imply good PER-QUERY ranking against a
  large same-category pool. The training diagnostic's near-zero gap is
  consistent with "the model satisfies the aggregate training objective
  well" without being consistent with "the model produces a fine-grained,
  reliable per-query ranking" -- and Recall@K tests exactly the latter.

**Put together: this project's frozen-SigLIP adaptation trained the
mechanism faithfully (see `architecture_notes.md`), but the repo's own
un-restricted in-batch negative sampling is a substantially easier
training signal than the category-restricted retrieval task actually being
evaluated, and the model appears to have found a solution that satisfies
that easier signal without transferring to the harder one.** This is not
proven with certainty (a full ablation re-training with category-restricted
in-batch negatives was not attempted -- out of scope for this phase, see
"Not attempted" below), but it is the most plausible, partially quantified
explanation available, and it is consistent with every piece of direct
evidence collected: the uneven published-ratio-by-K (a harder, top-heavy
metric like Recall@10 suffers disproportionately more than a looser one
like Recall@50, exactly what "the model never learned fine within-category
ordering" would predict), the below-raw-SigLIP result (raw visual
similarity is itself a directly useful within-category ranking signal that
a mechanism trained on a different, coarser task can end up discarding --
the same "learned projection throws away useful raw structure it never
needed to preserve" pattern this project first saw in phase 7), and the
confirmed absence of embedding collapse as an alternative explanation.

## The three-way conditioning-mechanism comparison (step 5's actual point), all on the identical frozen SigLIP backbone

| Mechanism | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| CSA-Net (category-pair subspace attention, phase 13b) | 0.0725 | 0.1393 | 0.1844 |
| OutfitTransformer (set-encoder + outfit token, this phase) | 0.0201 | 0.0588 | 0.0911 |

**Holding the backbone constant, CSA-Net's category-pair-conditioned
mechanism clearly and substantially outperforms OutfitTransformer's
set-encoder mechanism here** -- roughly 3.6x / 2.4x / 2.0x at
Recall@10/30/50 respectively. This is a real, useful finding on its own:
even though phase 13b found CSA-Net's mechanism did NOT decisively beat
this project's own phase 12c mechanism, it decisively beats OutfitTransformer's
mechanism under this project's specific training regime (frozen backbone,
in-batch negatives, no target-category conditioning). A plausible
contributor, consistent with the above: CSA-Net's mechanism is explicitly
category-pair-conditioned (it always knows which category pair it's scoring),
while this OutfitTransformer adaptation has no target-category conditioning
at all (confirmed faithful to the actual reference implementation, see
`architecture_notes.md`) -- so CSA-Net's architecture has a structural
advantage for a category-restricted retrieval task that OutfitTransformer's
own reference mechanism, as implemented in the repo actually available, does
not share.

## Where phase 12c now lands relative to BOTH published baselines (not just one)

| Phase 12c mode | vs. CSA-Net published (Recall@10/30/50 ratio) | vs. OutfitTransformer published (Recall@10/30/50 ratio) |
|---|---|---|
| Substitute (0.0667/0.1307/0.1734) | 80.7% / 83.4% / 82.9% | 69.6% / 72.8% / 78.9% |
| Blend, 0.5 (0.0893/0.1695/0.2255) | 108.0% / 108.2% / 107.8% | 93.2% / 94.4% / 102.6% |
| Complement (0.0971/0.1875/0.2471) | 117.4% / 119.7% / 118.2% | 101.4% / 104.4% / 112.4% |

**Phase 12c's complement mode now exceeds BOTH published literature
baselines at every K, and blend mode exceeds both at K=30/50 (and is within
7% of OutfitTransformer's published K=10)**, with the same protocol-fidelity
caveat used throughout this project (different candidate-pool construction,
not a byte-for-byte matched evaluation). Substitute mode falls short of
both, as it already did against CSA-Net alone in phase 13b -- unsurprising,
since substitute mode is deliberately trained to mimic raw visual similarity
rather than compatibility, and both published baselines are compatibility-
retrieval numbers. **This is a stronger, more complete statement of phase
12c's standing than phase 13b alone could make** (phase 13b only had one
literature anchor to compare against); it does not change the mixed,
not-a-clean-win framing phase 13b already established for CSA-Net's own
mechanism specifically, since that comparison is about mechanisms holding
backbone constant, not about literature-anchor ratios.

## What this does and doesn't establish

- **Does establish**: a faithful, verified (source-code-confirmed, not
  paper-summary-guessed) adaptation of OutfitTransformer's actual set-encoder
  mechanism, trained to genuine convergence on its own training objective,
  underperforms every other mechanism this project has evaluated on this
  harness -- including doing nothing at all (raw SigLIP). This is a real,
  useful negative result about this specific training regime (frozen
  backbone, in-batch non-category-restricted negatives, no target-category
  conditioning), not evidence that OutfitTransformer's architecture is
  fundamentally weak -- the published numbers this project is comparing
  against come from a full end-to-end CLIP fine-tune with 200 epochs and
  (per the paper, though not the specific repo implementation checked here)
  target-category conditioning, a substantially richer training signal than
  what was reproduced here.
- **Doesn't establish**: that a category-restricted in-batch negative
  sampling scheme, or a fine-tuned (non-frozen) backbone, would close this
  gap -- neither was tried here, consistent with the brief's own "do not
  attempt a full fine-tuned reproduction" instruction for this phase.
- **Doesn't establish** that CSA-Net's mechanism is "the right" conditioning
  approach in general -- only that, under this project's specific frozen-
  SigLIP/local-training constraints, it produced a mechanism that transfers
  to the actual (category-restricted) retrieval task far better than this
  particular OutfitTransformer adaptation did.

## Not attempted (per the brief's own scope)

No full fine-tuned, non-frozen-backbone reproduction of OutfitTransformer
was attempted (explicitly out of scope for this phase). No category-
restricted in-batch negative variant was tried to directly test this
phase's own explanation for the gap -- a reasonable, cheap next step if this
mechanism is revisited, but not attempted here since it would be a new
experimental variant beyond this phase's brief. No new baseline
reproductions or paper writeup were started, per the brief's explicit
"do not do yet."

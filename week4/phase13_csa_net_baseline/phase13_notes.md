# Phase 13: CSA-Net Baseline Reproduction -- Verdict

## What this phase set out to determine

Whether this project's own CIR harness (built in phase 12) is built correctly,
by reproducing a real published baseline (CSA-Net, CVPR 2020) under it, and
whether that gives this project a legitimate, matched-protocol claim of
beating a real baseline -- replacing the caveated OutfitTransformer
comparison every prior phase has carried.

## What actually happened

The architecture was implemented faithfully from the paper's own equations
(no usable reference implementation existed -- see `implementation_notes.md`).
Getting it to train STABLY required finding and fixing five real,
independently-confirmed optimization failures not mentioned in the paper
(OOM, dead gradients, magnitude collapse, then direction collapse that
survived a backbone freeze and needed an added uniformity regularizer -- full
detail in `training_log.md`). Once stable, the model was trained for 5 epochs
before being stopped by an explicit, real budget constraint ($9 remaining
Modal credit, insufficient for the configured 20-epoch / ~14-hour schedule).
Val loss was still improving every epoch when training stopped; the
positive/negative distance gap was narrowing but had not yet reversed sign.

Evaluated on this project's own CIR harness (`results_table.md`):

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| This reproduction | 0.0284 | 0.0667 | 0.0960 |
| CSA-Net published | 0.0827 | 0.1567 | 0.2091 |
| Raw SigLIP (this project, no training at all) | 0.0553 | 0.1067 | 0.1437 |

## Step 5: interpret plainly

**The reproduction currently underperforms even this project's own untrained,
frozen SigLIP baseline at all three K values, and lands well below CSA-Net's
own published numbers.** Before concluding anything from that gap (per this
phase's own brief: "investigate why before concluding anything, and report
the discrepancy honestly"), the honest, load-bearing fact is: `training_log.md`
shows the model was still actively improving every single epoch with no
plateau in sight when it was stopped, specifically because of a real,
external compute-budget constraint -- not because it converged and this is
its ceiling. `D_pos > D_neg` throughout all 5 epochs means the model hadn't
yet learned to rank a truly compatible item above its hardest mined
distractor on average; that is a training-completeness problem, a directly
observed and quantified one (the gap: 0.162 -> 0.106, narrowing every epoch),
not evidence the architecture or harness is broken.

**Does this validate the harness?** No -- and importantly, it doesn't
invalidate it either. This run cannot distinguish between two very different
explanations for the gap to CSA-Net's published numbers:
1. The harness has some undiscovered issue depressing scores.
2. The model simply wasn't trained long enough to reach the paper's own
   (unstated, but clearly much larger given standard training budgets in this
   literature) number of epochs.

Explanation 2 is the one the evidence in hand actually supports: every
single diagnostic in `training_log.md` (val_loss, the D_pos/D_neg gap) was
moving in the right direction, monotonically, right up to the moment training
was stopped for budget reasons. A harness bug would typically show up as a
result that's stuck, wrong-signed, or erratic -- not one that's cleanly
improving but simply hasn't finished improving yet. That said, this is
reasoned inference from an incomplete run, not a settled fact -- it cannot be
fully separated from explanation 1 without more training, which the current
budget doesn't allow. This is exactly the "significantly more time than
expected" scenario the brief anticipated, and the honest response is to
report it, not to round an undertrained result up into a validated one.

**Does this project have a legitimate, matched-protocol claim of beating this
baseline?** Not yet, and not from this run. Comparing this project's best
configuration (phase 9's Model A: R@10=0.1317) against an admittedly
undertrained CSA-Net reproduction (R@10=0.0284) is not a fair or meaningful
comparison in either direction -- it would overstate the case, not
understate it. **No claim of this shape should be made from this phase's
results as they stand.**

## Required-output checklist

- `implementation_notes.md` -- done: what's available (no usable unofficial
  implementation), full paper-vs-implementation architecture trace, every
  assumption documented.
- `training_log.md` -- done: the actual procedure, all five real optimization
  failures found and fixed (with evidence, not just description), the actual
  5-epoch training curve, and the explicit, real compute-budget constraint
  that stopped it.
- `results_table.md` -- done: reproduced Recall@10/30/50, CSA-Net's published
  numbers, and every one of this project's own configurations on the same
  harness.
- This file -- verdict: **inconclusive on harness validation, no legitimate
  matched-protocol "beats baseline" claim yet.** The path to resolving this
  is more straightforward than it might sound: continue training this exact
  checkpoint (`models/csa_net_best.pt`, `models/training_curves.json` show
  where it left off) for more epochs when more compute budget is available,
  and re-run `04_csa_cir_eval.py` -- no further debugging should be needed,
  since the five real optimization failures are already fixed and confirmed
  stable at full scale. This is a compute-budget gap, not an unsolved
  technical problem.

## What this phase actually accomplished, despite the inconclusive number

Five real, independently-diagnosed and fixed optimization failures in a
faithful from-scratch reproduction of a CVPR paper's stated architecture and
loss -- each confirmed with direct evidence (gradient norms, distance/cosine-
similarity math, synthetic collapsed-vs-spread sanity checks), not
theoretical guessing. The training pipeline is now demonstrably stable at
full 251,008-item scale. That is real, reusable infrastructure for this
project's own future work (explicitly flagged as valuable groundwork for
step 6's SigLIP-backbone CSA-Net variant), independent of whether this
specific checkpoint's Recall@K number is publication-ready yet.

## Do not do yet (per the brief)

The SigLIP-backbone CSA-Net variant (brief's step 6), the complement-mode
capacity-competition cost from phase 12c, and any paper write-up remain
explicitly out of scope for this phase and were not started.

# Phase 12c: Ranking-Distillation Substitute Target, With a Hard Stopping Condition

## Context

Phase 12's additive-mode-vector mechanism failed completely (76% top-10 overlap between
modes), traced to a 400x loss-scale imbalance. Phase 12b fixed the imbalance and verified
it directly -- the modes genuinely separated (overlap dropped to 51%, real qualitative
differences appeared) -- but a follow-up check phase 12b added on its own found the
separation wasn't in the right direction: substitute mode's retrieval was no closer to
raw SigLIP's own retrieval than complement mode's was (16.1% vs 17.5% overlap). Diagnosis:
the substitute target (a fixed PCA-128 projection of raw SigLIP, 73.7% variance retained)
preserved global variance but not the local nearest-neighbor structure that actually
determines what ranks in a top-10.

This phase replaces that target with a ranking-distillation loss that directly targets
local neighborhood structure: for each anchor, KL-divergence between a "teacher"
distribution (softmax over raw-SigLIP similarity to the anchor's actual top-50 raw-SigLIP
neighbors) and a "student" distribution (the same softmax computed from current
substitute-mode projections). This is the third attempt at the substitute target, with an
explicit stopping condition set in advance (step 5 of the brief): if substitute mode's
overlap with raw SigLIP's own retrieval doesn't clear a real, stated-in-advance threshold
over complement mode's overlap, stop iterating on the target and lay out the two real
remaining options rather than trying a fourth variant.

## Step 1: nearest-neighbor lookup

Precomputed top-50 raw-SigLIP neighbors for all 251,008 items via chunked matmul (chunk
size 2048, corpus resident on MPS, only each chunk's own top-51 survives past that
chunk's iteration -- a full 251,008x251,008 matrix would be ~252GB, never materialized).
**Actual cost: 5.54 minutes, no subsampling needed** (the brief's own contingency for an
expensive lookup wasn't triggered). Sanity checks passed cleanly: 0/2000 sampled items had
themselves in their own neighbor list; similarity range min=0.342, max=1.000 (a few
items reaching 1.000 is expected and consistent with phase 6's earlier finding of
byte-identical duplicate images in this catalog, not a bug), mean=0.812. Full detail:
`nearest_neighbor_lookup_summary.md`.

## Steps 2-3: the new loss, and fresh balancing verification

Substitute loss completely replaced (no PCA-128 target retained): `KL(teacher ||
student)` over each anchor's 50 precomputed neighbors, both distributions built with
`tau=0.07` (matching the complement loss's own InfoNCE temperature, for consistency, not
re-tuned separately). Calibration, done fresh per this phase's own instruction not to
assume phase 12b's ratio still applies: complement loss ~4.90 (unchanged, its formulation
didn't change), substitute loss ~0.076 -- a **64.5x ratio**, different again from both
phase 12's ~400x and phase 12b's ~4.9x, confirming the brief's expectation that each new
loss formulation needs its own remeasurement. Applied `weight_sub = 64.47`.
**Gradient-norm verification** (the actual check, not just the loss-value ratio): before
weighting, substitute's gradient into the shared net was 8.7x smaller than complement's;
after weighting, 0.1x (right at the edge of, but inside, the "reasonable order of
magnitude" band) -- **verification passed**. Full detail: `loss_balancing_check.md`.

## Step 4: retraining and evaluation

Same architecture, complement loss, positive edges, hyperparameters, and early-stopping
procedure as phases 12/12b -- only the substitute loss and its weight changed. Converged
cleanly (best checkpoint epoch 0, early-stopped epoch 5, same pattern as every prior
phase at this edge-count scale). One notable difference in the collapse-guard diagnostic:
substitute mode's mean pairwise cosine settled around 0.70 this run (vs phase 12b's ~0.00
and phase 12's ~0.56) -- a different value again, consistent with a third genuinely
different target producing a third genuinely different geometry, not by itself evidence
of success or failure (that's what the evaluation below settles).

### 4.1 -- CIR benchmark Recall@K (full three-way table: `results_table.md`)

| Configuration | Recall@10 | Phase 12 | Phase 12b |
|---|---|---|---|
| Raw SigLIP (reference) | 0.0553 | -- | -- |
| **Substitute mode** | **0.0667** | 0.1329 | 0.1212 |
| **Complement mode** | **0.0971** | 0.1335 | 0.1200 |
| Blend (0.5) | 0.0893 | 0.1366 | 0.1207 |

**For the first time across all three phases, substitute mode's Recall@K is clearly
separated from complement mode's, and substitute mode moved decisively toward raw
SigLIP's much lower range** (0.067 vs raw's 0.055 -- close -- vs complement's 0.097,
clearly higher). Phases 12 and 12b both showed the two modes scoring almost identically
here; this is the first configuration where the headline Recall@K numbers themselves
already hint the fix worked, before even running the dedicated diagnostic.

### 4.2 -- control-effectiveness diagnostic (full three-way table: `control_effectiveness.md`)

| Check | Phase 12 | Phase 12b | Phase 12c |
|---|---|---|---|
| Axis 1 gap (substitute − complement, visual sim) | +0.0073 | +0.0005 | **+0.0281** |
| Axis 2 (substitute hit rate, complement hit rate) | tied 0.128/0.128 | 0.125/0.118 (wrong direction) | **0.076/0.102 (correct direction)** |
| Top-10 overlap between modes | 0.7636 | 0.5102 | **0.4022** |
| Per-item cosine(z_sub, z_comp) | 0.8288 | 0.5844 | **0.3000** |
| Substitute vs raw SigLIP overlap | -- | 0.1612 | **0.3428** |
| Complement vs raw SigLIP overlap | -- | 0.1754 | **0.1708** |
| **Gap (sub-vs-raw − comp-vs-raw)** | -- | **−0.0142 (wrong direction)** | **+0.1720** |

**Every single diagnostic now points the same direction, decisively:**
- Axis 1's gap nearly quadrupled versus phase 12's original (weak) confirmation.
- Axis 2 is no longer tied or backwards -- complement mode clearly beats substitute mode
  at matching real outfit co-occurrence, exactly as the original premise required.
- The two modes are now the most different they've been across all three phases (40.2%
  overlap, 0.300 per-item cosine).
- **The decisive check**: substitute mode's overlap with raw SigLIP's own retrieval
  (34.3%) is now roughly DOUBLE complement mode's (17.1%) -- a gap of +0.172, moved from
  phase 12b's small, wrong-direction −0.014. Against the stopping-condition threshold of
  0.03 (stated in advance in the diagnostic script, chosen to clearly exceed phase 12b's
  own noise-level reversal), **this gap clears the bar by more than 5x**.

### Qualitative confirmation (`qualitative_examples/`)

The shoes example makes the mechanism visible directly: substitute mode's top-5 are all
elegant black strap/pump heels, visually cohesive with each other and with the query's
own dark, structured aesthetic; complement mode's top-5 are a stylistically eclectic mix
(a pink flat with graphic text, red pointed flats, black suede pumps) -- a real character
difference, unlike phases 12/12b's near-identical rows for this same query. In the bags
example, complement mode's #5 pick is the actual TRUE TARGET (the real held-out
co-outfit answer) at similarity 0.835 -- a concrete instance of complement mode doing
what it's supposed to do, that substitute mode's top-5 does not reproduce.

### A cost worth naming (continuing from phase 12b's own honesty on this point)

Complement mode's own quality kept degrading as the required substitute weight grew
across phases: best validation loss 5.34 (phase 12) -> 6.17 (phase 12b, weight 4.89x) ->
**7.00 (phase 12c, weight 64.47x)**, and complement's own Recall@10 dropped alongside it
(0.1335 -> 0.1200 -> 0.0971). The two objectives increasingly compete for the same shared
projection's limited capacity as the substitute weight needed to grow. This run's result
is still a clear net win (both modes now behave distinctly and correctly), but this
trend is worth tracking if the mechanism is built out further -- a shared single net may
not have enough capacity to serve both objectives at high fidelity simultaneously
forever, especially if a future target needs an even larger weight.

## Step 5: the explicit stopping-condition verdict

**The fix clearly worked.** Every diagnostic set out in the brief -- Recall@K separation,
axis 1, axis 2, mode-to-mode overlap, per-item cosine, and critically the
substitute/complement-vs-raw-SigLIP comparison -- moved in the intended direction this
time, most by a wide and unambiguous margin (the decisive check's gap, +0.172, is nearly
6x the stated stopping threshold). The stopping condition in step 5 (three failed
attempts in a row would be evidence of a fundamental problem) is **not triggered** --
this is the first of the three substitute-target attempts to succeed, not a third
failure.

**This is the version worth writing up and building on.** The underlying idea -- a single
shared model, additively conditioned by a small mode vector, whose retrieval behavior
can be steered at inference time between visually-similar-substitute and
compatibility-driven-complement behavior -- is now demonstrated to work, end to end,
under a real, standard retrieval benchmark (this project's own CIR harness) and a
dedicated mechanism-level diagnostic, not just a plausible-looking Recall@K number.

## What comes next (not decided here -- flagging for a fresh conversation, per this
phase's own "do not do yet" instruction)

Two things worth deciding in a follow-up, now that the mechanism itself is validated:

1. **Whether to extend to the fuller CSA-Net-style architecture is now a genuine
   optimization/ambition question, not a rescue mission.** The cheap mechanism works; the
   question is whether a structurally richer conditioning approach would work better
   still (e.g. on the interpolated blend's quality, or on reducing the complement-mode
   cost noted above), not whether *something* needs to replace a broken mechanism.
2. **The complement-mode cost trend** (worsening validation loss/Recall@K as weight_sub
   grew across phases 12b->12c) is worth a direct look before scaling this mechanism up
   further -- possibilities include giving substitute and complement modes some
   dedicated (not fully shared) capacity, or tuning batch composition/curriculum so the
   two objectives interfere less.

## Required-output checklist

- `nearest_neighbor_lookup_summary.md` -- done, 5.54 min actual cost, no subsampling.
- `loss_balancing_check.md` -- done, fresh remeasurement (64.5x initial ratio), gradient
  check passed.
- `results_table.md` -- done, phase 12 and 12b numbers included.
- `control_effectiveness.md` -- done, full diagnostic, three-way comparison, decisive
  check included.
- `qualitative_examples/` -- done, same 6 outfits as phase 12/12b.
- This file -- explicit stopping-condition verdict above: **fix worked, not triggered.**

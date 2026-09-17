# Phase 32: Honest Interpretation

## Exactly how many seeds got trained, and why

**3 seeds total: 42 (reused from phase 31 at zero cost), 1, and 2 (both trained this phase).** The budget
constraint that stopped phase 31 early did NOT bind here -- both planned seeds trained successfully within
budget (total phase 32 spend: $0.88, final remaining balance: $6.12, never dropping below the $1.50 safety
margin). This phase reached its full intended scope (per the brief's own step 2: "train two additional
seeds," not "train as many seeds as budget allows") without needing to stop short. Reported exactly as it
happened: this is a genuine 3-seed ensemble by design, not a truncated attempt at a larger one. See
`budget_tracking.md` for the full balance-check log.

## Does the partial ensemble close, partially close, match, or exceed the remaining 5.5% gap to phase 28?

**It closes almost all of it, but does not cross it, at any K.** Exact numbers, no rounding in either
direction:

| Configuration | R@10 | R@30 | R@50 |
|---|---|---|---|
| Phase 31 single model | 0.1799 | 0.3111 | 0.3844 |
| **Phase 32, 3-seed ensemble** | **0.1897** | **0.3246** | **0.4019** |
| Phase 28, 10-model ensemble | 0.1904 | 0.3267 | 0.4079 |

Phase 32's 3-seed ensemble reaches **99.6% / 99.4% / 98.5%** of phase 28's 10-model ensemble -- it does
**not** beat phase 28 at any K (0.1897 < 0.1904, 0.3246 < 0.3267, 0.4019 < 0.4079, all three margins real
and consistent, not noise-level ambiguity in one direction). This is neither "the gap closed" nor "our
model's lead held up strongly" -- it is a near-exact tie, with phase 28 still narrowly ahead at every K,
achieved using less than a third of phase 28's own seed budget (3 vs. 10). Framed the other way: phase 31's
single model closed the OutfitTransformer-vs-project gap from 30.9% (phase 14b's original number) to
94.5-95.2% of phase 28's ensemble; this phase's partial ensemble closes it further, to 98.5-99.6%, still
short but by a genuinely small, honestly-reported margin.

## Does OutfitTransformer's ensembling gain look proportionally similar, smaller, or larger than this project's own pattern?

Ensembling gain over the best individual seed, at R@10:

| Mechanism | Seeds | Gain over best solo seed |
|---|---|---|
| Phase 26 (this project's own model, image-only) | 10 | +17.4% |
| Phase 28 (this project's own model, text-only) | 10 | +7.7% |
| **Phase 32 (OutfitTransformer, this phase)** | **3** | **+4.8%** (val: 0.1942 -> 0.2035) |

**Smaller in absolute terms, but not directly comparable given the seed-count mismatch (3 vs. 10) --
stated honestly as an open question, not resolved here.** Phase 26/28's own ensemble-size sweeps
(`week4/phase26_ensembling/ensemble_size_sweep.md`, `week7/phase28_text_ensemble/ensemble_size_sweep.md`)
both show most of the total 10-seed gain arriving within the first 2-5 members, with diminishing but real
returns continuing to size 10 -- so a fair reading of phase 32's +4.8% at size 3 is that it likely
represents most, but probably not all, of what a full 10-seed OutfitTransformer ensemble would eventually
reach. The individual seeds' unusually tight spread here (std 0.00074, vs. phase 26/28's 0.0014-0.0016 --
see `individual_seeds.md`) is a plausible partial explanation: less per-query disagreement between seeds
gives score-averaging less to correct, consistent with this project's own established finding (phase 26's
own notes) that tight seed-score variance doesn't guarantee ensembling payoff, but it doesn't rule out a
larger ensemble finding more disagreement to exploit either -- both directions are genuinely untested here.

**The honest, most defensible statement**: with only 3 of a possible 10 seeds, OutfitTransformer's
ensembling gain is not yet distinguishable from "on track to match phase 28's proportional gain" versus
"structurally smaller" -- both are consistent with the data actually collected. This is exactly the kind
of claim this phase's brief asked not to round in either direction, and it is not rounded here.

## The honest, final comparison number for this project going forward

**Phase 32's 3-seed ensemble -- test-benchmark Recall@10/30/50 = 0.1897/0.3246/0.4019, checkpoints
`week7/phase32_partial_ensemble_outfittransformer/models/ot32_seed{42,1,2}.pt` (score-averaged, never
embedding-averaged) -- is now the honest OutfitTransformer baseline citation for this project, superseding
phase 31's single-model number (0.1799/0.3111/0.3844), which is itself superseded from phase 14b's
original (0.0588/0.1286/0.1809).** Each prior number remains valid and citable in its own role in the
progression (un-invested single config -> fully-tuned single model -> partially-ensembled), none are
deleted or invalidated.

**Phase 28's mean-pooled text-only 10-model ensemble (Recall@10/30/50 = 0.1904/0.3267/0.4079) remains this
project's best overall result, ahead at every K, by a margin of 0.4-1.5% relative.** This is a real,
measured lead, not a rounding artifact -- but it is now a genuinely close comparison, not the 3x-plus gap
implied by earlier framings (phase 14b's original number vs. phase 28: OutfitTransformer captured only
30.9% of phase 28's result; now, at 3 seeds, it captures 98.5-99.6%). Any report language should state
this precisely: **our model's advantage over a fairly-invested OutfitTransformer reproduction, at matched
(if partial, for OutfitTransformer) ensembling investment, is now under 2% relative at every K** -- a
categorically different claim than "our approach substantially outperforms OutfitTransformer," which the
project's earlier, under-invested baseline comparison would have supported but this phase's evidence does
not.

## What remains genuinely open

Whether a full 10-seed OutfitTransformer ensemble (this project's own established scale for a "complete"
ensembling investment) would close the remaining <2% gap, match it exactly, or exceed it is **not
determined by this phase** -- it is close enough that either outcome is plausible given the numbers
actually measured, and stating a prediction here would be exactly the kind of unearned rounding the brief
warned against. `scripts/02_train_seed.py` is ready to train seeds 3-9 the moment Modal budget allows,
using the identical winning configuration and the same sequential, balance-checked launch pattern proven
safe in this phase.

Architectural scale testing (phase 31's step 4, never run) also remains completely untested for
OutfitTransformer -- whether widening the transformer itself (independent of ensembling) would move this
comparison further is a separate, still entirely open question.

# Phase 31: Honest Interpretation

## Scope disclosure, up front: this phase stopped early, for budget reasons, not methodological ones

Steps 1-3 (text input, checkpoint-selection fix, hyperparameter tuning) completed in full, exactly as
planned. Steps 4 (architectural scale testing) and 5 (10-seed ensembling) were **not completed** -- this
project's Modal account hit a hard budget wall mid-phase. A billing check
(`modal billing report --for "this month"`), prompted directly by the user flagging "$7 left," showed
**$22.28 already spent on phase 31 alone** against **~$7 remaining** in the account. The 30-config scale
sweep (`04_scale_sweep.py`, already launched and dispatched to Modal) was stopped immediately
(`modal app stop --yes`) before any config completed -- nothing was checkpointed (these were throwaway
sweep configs, `save_checkpoint=False`), so nothing was lost, but nothing was gained either. Step 5 was
never started. Step 6 (the final test-benchmark evaluation) still ran, but on a **single model**, not an
ensemble -- see `scale_sweep.md`, `individual_seeds.md`, `ensemble_size_sweep.md` for the full disclosure
of what was skipped and why.

This is reported directly rather than silently absorbed or hidden behind partial numbers, per this
project's standing honesty convention. What follows is a genuine, complete result for the steps that did
run, with the missing steps' likely effect discussed honestly (not guessed at as if it were measured).

## Benchmark-discipline correction

Phase 14b's reported `0.0588/0.1286/0.1809` was measured against `week4/phase12_controllable_modes/data/
cir_benchmark.json` -- the file every phase from 23 onward treats as the **test** benchmark, touched
exactly once. Phase 14b had no separate validation benchmark and made no tuning decisions with it, so it
never violated a rule that didn't yet exist for it -- but its number is a test-set measurement, and this
phase's own validation-side numbers (A0-A3, the tuning sweep) must never be directly compared to it.
`cir_val_benchmark.json` (built later, in phase 23) is what every selection/tuning decision in this phase
used instead. `train_one` never loads the test benchmark at all -- verified by grep, not just convention.

## Per-step contribution progression

Validation-side (selection signal, used for every decision through step 3) and test-side (the one final,
honest number) are kept visibly separate, since they were never meant to be compared directly:

| Step | Configuration | Val R@10 | Relative gain |
|---|---|---|---|
| A0 | Phase 14b as-shipped, evaluated on val for the first time | 0.0659 | -- |
| A1 | Modal port, val_loss selection (fidelity check) | 0.0637 | -3.3% (reproduction noise, not a regression to explain) |
| A2-corrected | + Recall@10 selection (patience recalibrated to 25) | 0.0786 | +23.4% over A1 |
| A3-corrected | + text input | 0.0890 | +13.2% over A2-corrected |
| Step 3 final | + LR/batch-size tuning + loss-shape tuning (uniformity_weight, margin) + full training budget | **0.1924** | **+116.2% over A3-corrected** |
| Step 4 | Architectural scale testing | NOT RUN (budget) | -- |
| Step 5 | 10-seed ensembling | NOT RUN (budget) | -- |

**No step is credited with more than it earned.** A1 vs. A0 is a fidelity check, not a "gain" -- a Modal
reimplementation landing within 3.3% of the original on a benchmark it was never tuned against is a clean
reproduction, reported as such rather than rounded into the progression's gain chain. The single largest
lever in the entire phase was step 3's loss-shape tuning, specifically correcting `uniformity_weight` from
phase 14b's inherited 1.0 down to 0.1 -- see `tuning_log.md` for the full breakdown; that one change alone
took val Recall@10 from 0.0757 to 0.1850 (+144% relative) at fixed LR/batch size, and is directly
mechanistically connected to the checkpoint-selection finding: the uniformity term wasn't just corrupting
the OLD selection signal, it was actively suppressing training itself at its inherited weight.

## Test-side result and the gap verdict

Single model (step 3's winner, no ensemble), evaluated once on the real test benchmark:

| Configuration | R@10 | R@30 | R@50 |
|---|---|---|---|
| Phase 14b original (single config) | 0.0588 | 0.1286 | 0.1809 |
| OutfitTransformer published | 0.0958 | 0.1796 | 0.2198 |
| **Phase 31 strengthened, single model** | **0.1799** | **0.3111** | **0.3844** |
| Phase 28 text ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 |

- **vs. phase 14b's original: +206.0%/+141.9%/+112.5% relative (3.06x/2.42x/2.13x).**
- vs. OutfitTransformer's published numbers: 187.8%/173.2%/174.9%.
- **vs. phase 28's own 10-model ensemble: 94.5%/95.2%/94.3%** -- a single, un-ensembled model reaches
  within ~5-6% of the project's own best, fully-invested, ensembled result.

Per the brief's own three-way framing (closes / partially closes / doesn't move meaningfully): **the gap
closes dramatically, though not completely, and it closes with a single model against the project's own
ensemble** -- an asymmetric comparison that still favors this reading, not the reverse. At phase 14b's
original number (0.0588), OutfitTransformer's mechanism captured only 30.9% of phase 28's ensemble result;
fairly invested (text, correct checkpoint selection, tuned hyperparameters), it captures 94.5%, without
even the scale testing or ensembling this phase couldn't afford to run.

## What this suggests was driving the earlier lead

**Overwhelmingly, it was investment gap, not a fundamental mechanism-quality difference.** Three concrete,
diagnosed, quantified reasons, all specific to how OutfitTransformer's reproduction had been built and
evaluated, none about the transformer set-encoder mechanism itself being weak:

1. **Checkpoint selection was broken in a way that mattered a lot**: `checkpoint_selection_check.md` shows
   phase 14b's triplet margin was never satisfied across its entire 92-epoch run, and val_loss "improved"
   almost entirely via the uniformity regularizer term instead. The selected checkpoint reflected embedding
   spread, not ranking quality.
2. **The uniformity regularizer's weight (inherited, never tuned) was itself actively harmful**: dropping
   it from 1.0 to 0.1 alone (holding LR/batch size fixed) produced a +144% relative val Recall@10 gain --
   the single largest lever in this phase, bigger than text, bigger than LR/batch-size tuning, bigger than
   the selection fix itself.
3. **No text input, and no hyperparameter tuning at all** -- phase 14b's `lr=2e-5` (inherited from a
   reference repo that fine-tunes a full backbone, not a small frozen-backbone head) turned out to be far
   from optimal (`lr=1.5e-4`, 7.5x higher, won the tuning grid).

None of these three findings say anything about whether a multi-head self-attention set-encoder is
inherently a better or worse complementary-retrieval mechanism than this project's own mean-pooled
projection -- they say phase 14b's specific implementation of that mechanism was under-invested in exactly
the ways this phase's brief predicted, and correcting them closed most of the previously-reported gap.

## Untested but suggestive: what ensembling would likely have added

This is stated as a reasoned expectation, not a measured result, and is clearly labelled as such. Phase 26
(image-only) and phase 28 (text-only) both saw real, meaningful gains from 10-seed score-averaging
ensembling over their own respective single-model results: +17.4%/+11.5%/+9.3% (phase 26) and
+7.7%/+7.0%/+6.6% (phase 28) relative, respectively, at R@10/30/50. If phase 31's single model (0.1799 at
R@10) saw a similar proportional ensembling gain, an ensembled version could plausibly land in the
0.19-0.21 range at R@10 -- at or above phase 28's own 0.1904. This is exactly the kind of result that would
need to be measured, not assumed, before it becomes a citable number -- flagged here as the clear, obvious,
well-motivated next step if Modal budget becomes available again, not claimed as this phase's own finding.

## Overfitting evidence

Not applicable -- step 4 (where the train-vs-val Recall@10 diagnostic would have been reported) was not
run. `cir_train_benchmark.json` was uploaded to the Modal volume and `train_one` computes the diagnostic
whenever `eval_train_benchmark=True` (the default), but no scale-varying runs exist to compare it across.

## The honest, final baseline number for this project going forward

**Phase 31's strengthened single model -- test-benchmark Recall@10/30/50 = 0.1799/0.3111/0.3844,
checkpoint `week7/phase31_fair_baseline_outfittransformer/models/ot31_budget_check_full.pt` -- is now the
honest OutfitTransformer baseline citation for this project.** Phase 14b's 0.0588/0.1286/0.1809 is
superseded as *the* baseline number -- any future "our model beats OutfitTransformer" claim must cite
phase 31's number, not phase 14b's, and must state plainly that this is a single model, not the fully
scale-tested and ensembled result the brief originally called for, because that additional investment was
not completed due to a real budget constraint, not because it was judged unnecessary. Phase 14b's own
number remains valid and citable specifically as "the un-invested, single-configuration starting point,"
exactly the role it plays in the progression table above -- it is not deleted or invalidated, just
superseded as the baseline any comparison should actually use.

**Phase 28's mean-pooled text-only ensemble (Recall@10/30/50 = 0.1904/0.3267/0.4079) remains this
project's best overall result.** It still beats phase 31's single model at every K, by a real but now much
smaller margin (5.5%/4.8%/5.8% relative) than the margin against phase 14b's original number (3.06x). Given
the untested-but-suggestive ensembling projection above, whether this margin would survive a properly
resourced phase 31 (scale testing + ensembling) is a genuinely open, well-motivated question for a future
phase, not a settled one either way.

## For the report

Per the brief's own instruction, the project's comparison framing should be updated to reflect: (1) the
matched-backbone reproduction gap between this project's model and OutfitTransformer's mechanism is now
known to be driven overwhelmingly by prior investment asymmetry, evidenced directly and quantitatively
here, not asserted; (2) the current honest comparison, at matched (if incomplete, for OutfitTransformer)
investment, is 0.1799 vs. 0.1904 at R@10 -- close, not the 3x-plus gap earlier framings implied; (3) this
narrowing came from a single model on the OutfitTransformer side against a fully-ensembled model on this
project's side, an asymmetry that itself should be named plainly in any report language, not smoothed over.

# Phase 29: Honest Interpretation

## A correction to this phase's own brief

The brief's Context and Step 5 sections cite phase 28's text-only ensemble as Recall@10/30/50 = 0.1888/0.3229/0.4028. Phase 28's actual verified numbers (`week7/phase28_text_ensemble/final_evaluation.md`, `phase28_notes.md`) are **0.1904/0.3267/0.4079**. This document and every other file in this phase uses phase 28's real, verified numbers throughout, mirroring how phase 28 corrected phase 27's brief.

## Headline: the single-seed gate says no, clearly, and the phase stops here

Per the brief's own explicit "stop early if no signal" instruction (Step 3), a single seed is trained and checked against phase 28's single-seed text-only result before any ten-seed budget is spent.

| Configuration | Val Recall@10 |
|---|---|
| Phase 28 text-only, single seed (42), mean-pool | 0.1819 |
| Phase 29 cross-attention, single seed (42) | 0.1228 |

That is not "no gain" -- it's a **32.5% relative regression**, far below the gate's required +2% to proceed. Per the brief's own instruction ("If it's the same or worse, stop before spending the ten-seed budget, report the result, and treat this as the answer for now"), this phase does not train the remaining 9 seeds, does not run an ensemble size sweep, and does not touch the test benchmark. `individual_seeds.md`, `ensemble_size_sweep.md`, and `final_evaluation.md` are intentionally not produced -- they're conditional on reaching full-ensemble scale in the brief's own required-outputs list, and this phase never gets there.

## What the learned attention weights actually look like, and why that matters here

This is not a case of the inductive bias failing to engage. `training_log.md`'s per-epoch attention entropy never approaches either failure mode the brief warned about: normalized entropy sits at 0.78-0.85 throughout training (1.0 would be uniform/mean-pool-equivalent, 0.0 would be full collapse onto one item). `attention_qualitative.md`'s direct inspection confirms this: attention weights over context items visibly, sometimes dramatically, shift depending on which candidate conditions them. In query 7064, for example, one context item ("sterling silver open[work wolf band]") gets weight 0.08 when the true target conditions attention, and weight 0.71 when a random negative bag conditions it instead. The mechanism is doing real, non-trivial, candidate-dependent work.

The problem is what it learned to do. Training data conditions attention on exactly one candidate per example: the true target. `context_query`/`candidate_key` therefore only ever learned to produce weightings that are good *for that specific true positive*, never for an arbitrary wrong candidate. At evaluation time the model must condition on the full negative-heavy pool (up to 3,000 candidates per category, the overwhelming majority of which are not the target) to do its actual job: rank the true target above them. The qualitative examples show exactly this -- attention conditioned on random negatives looks reactive and inconsistent rather than semantically grounded, swinging by 5-10x on the same context item depending on which unrelated candidate happens to be doing the conditioning. `training_log.md`'s curve shows the mechanism: training loss falls smoothly through epoch 6 while validation Recall@10 peaks at epoch 2 and degrades afterward -- the model keeps getting better at the training-time task (attend well when conditioned on the true positive) while getting worse at the actual evaluation task (attend usefully when conditioned on whatever candidate is being scored, most of which are wrong).

## Is this "mean-pooling is the right inductive bias" or "the cross-attention design had an implementation issue"? Neither, cleanly -- it's a diagnosed training/evaluation mismatch

This result does not cleanly support either of the brief's two framings. It is not evidence that mean-pooling is intrinsically the correct inductive bias for this task -- the underlying idea (some context items should matter more than others when scoring a specific candidate) was never actually given a fair test, because the training signal only ever taught the model to condition on one candidate. It is also not a plain implementation bug -- the math was checked directly (the evaluation path's einsum formulation was verified term-by-term against the training path's `query_for_candidate` method, and both apply the exact formula the brief specifies), and the qualitative check confirms the mechanism is functioning as designed, just poorly generalizing.

The honest characterization is the one `architecture_notes.md` flagged before any training ran: this phase's training procedure conditions attention on the true positive only, as a stated efficiency-preserving simplification versus full per-candidate conditioning (which would need O(batch_size^2) attention computation per step instead of O(batch_size), a genuine architecture/training-loop change beyond what "minimum viable inductive bias, single-variable test" calls for). That simplification turned out to matter more than expected -- the model overfits its conditioning behavior to the one candidate role it's trained against, and that specialization actively hurts, rather than merely fails to help, once it has to condition on the population of candidates it actually sees at evaluation time.

A genuinely fair test of candidate-conditioned cross-attention would need to condition training on a broader population of candidates (at minimum the batch's own negatives, not just the true positive), which crosses from a small architectural tweak into a real training-loop redesign -- outside the "one obvious fix, don't turn this into an open-ended tuning phase" allowance the brief grants for cases like adding LayerNorm. Per that same instruction, this phase reports the finding rather than attempting that redesign here.

## Is the gain large enough to justify the complexity? Not applicable -- there is no gain

The brief's final honest-interpretation question ("if it improves, is the gain large enough to justify the added complexity") doesn't apply: cross-attention did not improve on mean-pooling, at the single-seed level, by a wide margin. Mean-pooling remains this project's reference query-aggregation mechanism.

## Final, honest verdict

**Phase 28's mean-pooled text-only ensemble (Recall@10/30/50 = 0.1904/0.3267/0.4079) remains this project's best result.** Candidate-conditioned cross-attention, as implemented and trained in this phase, underperforms mean-pooling substantially at the single-seed level (0.1228 vs. 0.1819, -32.5% relative) and does not pass the brief's own gate to proceed to ensemble scale. The mechanism itself works as designed (non-collapsed, genuinely candidate-dependent attention, confirmed both quantitatively via entropy tracking and qualitatively via direct inspection) -- what fails is the training-time simplification of conditioning only on the true positive, which the model exploits by overspecializing rather than learning a generalizable weighting. This is the "well-motivated architectural change that didn't help" outcome the brief itself named as a valid, paper-worthy finding, with the added value of a specific, evidence-backed mechanism (training/evaluation conditioning mismatch) rather than a vague "it just didn't work."

This diagnosis also names a concrete, scoped follow-up for a future phase, should the project want to revisit this: training-time attention conditioned on more than the single true positive (e.g., a handful of the batch's own in-batch negatives, not the full O(B^2) candidate set) would directly test whether the underlying inductive bias has value once the training/evaluation mismatch identified here is closed -- not started in this phase, per its own scope discipline.

## Reference configuration, for the record

- **Architecture**: phase 28's projection head (1536 -> 1024 -> 128) unchanged, plus four new 128->128 linear layers (`candidate_key`, `context_query`, `context_value`, `candidate_value`; 66,048 parameters), replacing mean-pooling with candidate-conditioned cross-attention per the brief's exact formula.
- **Training data**: multi-item context, rebuilt directly from the official Polyvore `nondisjoint` outfit files (284,767 train / 26,781 val examples) rather than phase 27/28's flattened pairwise edges -- necessary because single-item context gives `candidate_key`/`context_query` provably zero gradient (see `architecture_notes.md`).
- **Learning rate**: 0.0005 (chosen over phase 28's 0.001 via a short 3-way check, see `training_log.md`).
- **Result**: single-seed (42) val Recall@10 = 0.1228, best at epoch 2, vs. phase 28's 0.1819. Gate: NO-GO. No ensemble trained, no test-benchmark evaluation performed.
- **Checkpoints**: `week7/phase29_cross_attention/models/xattn_seed42.pt` (projection head), `xattn_seed42_xattn.pt` (the four attention layers) -- kept for reference/reproducibility, not a recommended production configuration.

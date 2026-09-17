# Phase 36: Paper Prerequisites -- Notes (written as built)

Started 2026-09-15. Executes the "Outstanding work before drafting" list in `paper/paper_framing_direction.md`:
(1) equivalence testing for Claim 1, (2) the official released-split AUC/FITB evaluation of the three
matched-condition models, (3) measured inference cost, (4) parameter-ratio verification, (5) the writing
deliverables that depend on data already in hand (fidelity taxonomy, parameter-span figure).

## Test-benchmark discipline statement

This phase trains nothing and makes no selection decision of any kind. Every checkpoint it loads was finalized
in a prior phase and has already been evaluated on the test benchmark once, in that phase:

| System | Checkpoints | Test result already recorded in |
|---|---|---|
| `ours_ens` | `week7/phase28_text_ensemble/models/text_ensemble_seed{42,1..9}.pt` | phase 28 `final_evaluation.md` |
| `ours_solo` | `week7/phase27_text_and_category/models/text_only.pt` | phase 27 `results_table.md` |
| `ot_ens` | `week7/phase32_.../models/ot32_seed{42,1,2}.pt` | phase 32 `data/final_test_result.json` |
| `ot_solo` | `ot32_seed42.pt` (byte-identical to phase 31's `ot31_budget_check_full.pt`) | phase 31 `data/final_test_result.json` |
| `csa_ens` | `week7/phase34_.../models/csanet34_seed{42,1,2}.pt` | phase 34 `data/final_test_result.json` |

Re-scoring them here only adds per-query granularity to numbers that already exist. **The guard is that every
re-scored aggregate must reproduce its recorded value to 4 decimals before any new analysis is trusted.** No
CSA-Net solo model is scored (its seed-42 checkpoint was never test-evaluated; doing so would be a new touch of
a non-final checkpoint).

The official Polyvore `compatibility_test.txt` and `fill_in_blank_test.json` reference exactly the 10,000
`nondisjoint/test.json` outfits from which the CIR test benchmark was built (the CIR benchmark uses 9,786 of
them; verified during planning: 0 unresolved references, 0 items missing from the embedding matrix). Scoring
them is therefore not new data exposure -- it is a second protocol over the same released test split.

## Pre-declarations (committed BEFORE any script in this phase was run)

1. **Equivalence bound.** delta = 2% relative of `ours_ens`'s Recall@K at each K (approx. +/-0.0038 at K=10,
   +/-0.0065 at K=30, +/-0.0082 at K=50). Rationale: this project's own single-seed-gate convention (phases
   29/30 treated -2% relative as "same or worse, stop"). Two systems are declared equivalent at K iff the 90%
   paired-bootstrap percentile CI of their Recall@K difference lies entirely within [-delta, +delta] (two
   one-sided tests at alpha=0.05). The 95% CI is also reported. Paired bootstrap over query indices,
   B=10,000, seed 20260915.
2. **Official-split aggregation.** Primary protocol = "CIR-native leave-one-out": each model is scored through
   its own complementary-retrieval function, the thing the paper compares.
   - Outfit compatibility score = mean over items i of s(context = outfit minus {i}, candidate = i).
   - FITB = argmax over the 4 candidates of s(question, candidate); correct iff argmax is `answers[0]`.
   - s is: ours -> cosine(normalized mean of projected context, projected candidate); OutfitTransformer ->
     embed_query(context) . embed_item_alone(candidate); CSA-Net -> negative length-normalized average
     conditioned squared distance (paper eq. 5, exactly as `train_core.py`'s evaluator computes it), with the
     candidate's own category as the target category. Ensembles average member scores (cosines, or distances
     for CSA-Net), never embeddings -- identical to each phase's own CIR evaluator.
   - Secondary protocol = phase 9's exact protocol (mean pairwise cosine of unconditioned item embeddings;
     FITB = mean cosine to question items; strict `>` tie rule kept for reproduction), run for ours and
     OutfitTransformer (solo + ensemble) and for untrained references. Not applicable to CSA-Net, which has no
     unconditioned item embedding -- stated, not fudged.
   - Guards, run first: phase 9 Model A must reproduce AUC 0.9469 / FITB 0.7031 and raw image-only SigLIP
     must reproduce 0.7172 / 0.4843 under the secondary protocol.
3. **The official-split result is reported whichever way it lands.** Divergence from the CIR result would
   narrow Claim 1 and is publishable; it is not grounds for omission.
4. **Inference cost is reported whichever way it lands.**

## Deviations from the approved plan (running log)

- The plan said to *import* the phase 9 protocol functions. `06_evaluate_official.py` does a bare
  `from model import ProjectionHead` at import time, and phase 34's `train_core.py` does a bare
  `from model import CSANetSigLIP` -- with three different `model.py` files in play, a bare import can
  silently bind the wrong one. The protocol functions (resolver, rank-sum AUC, compatibility, FITB) are
  therefore re-typed verbatim in `scripts/common.py`, and the two guards in pre-declaration 2 validate the
  copy against phase 9's recorded numbers before anything new is computed. Model classes are loaded via
  `importlib` under distinct module names (phase 17's pattern).

## Results (filled in as each step completes)

### Step 1 -- per-query rank capture and the reproduction guard (`01_capture_per_query_ranks.py`, 68 s wall)

All five systems reproduced their recorded test aggregates. The three with exact JSON floats (`ot_ens`,
`ot_solo`, `csa_ens`) matched bit-for-bit (|diff| < 1e-9 at every K); the two recorded only to 4 decimals
(`ours_ens`, `ours_solo`) matched at 4 decimals. `n_total = 29,681`, `n_skipped = 0` for every system. The
copied scoring paths are therefore faithful, and the per-query ranks (`data/per_query_ranks_<system>.npz`) can be
trusted for everything downstream. Details: `data/reproduction_check.json`.

### Step 2 -- paired bootstrap equivalence test (`02_equivalence_test.py`; full tables in `equivalence_test.md`)

Headline, `ours_ens` minus `ot_ens` (pre-declared delta = 2% relative = +/-0.0038 / 0.0065 / 0.0082):

| K | diff | diff % of OT | 95% CI | 90% CI | TOST | McNemar p |
|---|---|---|---|---|---|---|
| 10 | +0.0007 | +0.36% | [-0.0030, +0.0043] | [-0.0025, +0.0037] | **equivalent** | 0.735 |
| 30 | +0.0022 | +0.66% | [-0.0021, +0.0065] | [-0.0014, +0.0057] | **equivalent** | 0.323 |
| 50 | +0.0060 | +1.49% | [+0.0016, +0.0103] | [+0.0024, +0.0097] | not equivalent | 0.0077 |

**Reading.** At K=10 and K=30 the two mechanisms are equivalent under the pre-declared bound and no difference is
detectable (McNemar p = 0.74 / 0.32; discordant pairs 1590/1570 and 2066/2002 -- essentially symmetric). At
K=50 there is a small, real advantage for ours: +1.49% relative, 95% CI excluding zero, McNemar p = 0.008. TOST
fails there not because the point estimate exceeds 2% (it does not) but because the upper end of the 90% CI
(+0.0097, i.e. 2.4% relative) does. The precise statement for the paper: *equivalent at K=10 and K=30; at
K=50 ours is ahead by a small margin whose plausible range (roughly +0.4% to +2.6% relative) is not narrow
enough to declare it within the 2% bound.* Not "comparable at every K"; not "ours wins."

The test discriminates where it should: `ours_ens` and `ot_ens` each beat `csa_ens` by +12% to +14% relative at
every K with CIs far from zero (McNemar p < 1e-26), so the equivalence at K=10/30 is not an artifact of an
insensitive test. Single models are NOT equivalent: `ours_solo` trails `ot_solo` by -8.0% / -5.3% / -2.9%
relative (all CIs exclude zero). The transformer is ahead at the single-model level; ensembling closes it (ours
gains +14.9% at R@10 from solo to 10 seeds, OutfitTransformer +5.4% from solo to 3 seeds).

Per category (R@10, `ours_ens` minus `ot_ens`): 9 of 11 CIs include zero; two favour ours significantly (hats
+0.022, sunglasses +0.014); none favours OutfitTransformer significantly; signs are mixed (5 positive, 6
negative). The two mechanisms trade categories with no systematic pattern -- consistent with equivalence rather
than with one mechanism dominating a subset of categories.

### Step 3 -- official released-split evaluation (`03_official_auc_fitb.py`, `03b_official_bootstrap.py`; tables in `official_benchmark.md`, `official_bootstrap.md`)

Both guards reproduced exactly (raw image-only SigLIP 0.7172 / 0.4843; phase 9 Model A 0.9469 / 0.7031), so the
re-typed protocol code is faithful. 0 of 20,000 compatibility lines and 0 of 10,000 FITB questions skipped;
107,012 leave-one-out tasks; 9,987/10,000 FITB questions have all four answers in one category.

Primary protocol (CIR-native leave-one-out), with published nondisjoint numbers for context:

| System | AUC | FITB |
|---|---|---|
| Raw SigLIP, image only (untrained) | 0.6870 | 48.43% |
| Raw SigLIP, image + text (untrained) | 0.6420 | 43.24% |
| `ours_solo` | 0.9567 | 72.95% |
| **`ours_ens`** | **0.9655** | **75.48%** |
| `ot_solo` | 0.9313 | 73.16% |
| `ot_ens` | 0.9349 | 74.64% |
| `csa_ens` | 0.9349 | 70.51% |
| *Published* Vasileva et al. 2018 (ResNet-18 fine-tuned) | 0.88 | 57.6% |
| *Published* CSA-Net 2020 (ResNet-18 fine-tuned, no text) | 0.91 | 63.73% |
| *Published* OutfitTransformer 2023 (ResNet-18 fine-tuned + SentenceBERT fc) | 0.93 | 67.10% |

Paired bootstrap (`official_bootstrap.md`; the 2% rule is applied here **post hoc**, labelled as such):

- `ours_ens` - `ot_ens`: **AUC +0.0307**, 95% CI [+0.028, +0.033] -- a clear, real gap, not equivalent by any
  reading. **FITB +0.84 pts**, 95% CI [+0.13, +1.58], McNemar p = 0.026 -- a small, significant advantage for ours
  that sits inside the post-hoc 2% band.
- `ot_ens` - `csa_ens`: AUC **+0.0000**, 95% CI [-0.0024, +0.0023] -- the two baselines are indistinguishable on
  AUC; FITB +4.1 pts (OT ahead, p < 1e-25).
- `ours_ens` - `csa_ens`: AUC +0.0307, FITB +5.0 pts, both far from zero.
- `ours_solo` - `ot_solo`: AUC +0.0255 (ours ahead); FITB -0.21 pts, p = 0.63 (tie).

**Reading, per pre-declaration 3 (reported whichever way it lands).** Two findings, one of which narrows Claim 1.

1. **All three matched-condition frozen-backbone systems exceed all three published fine-tuned-backbone numbers on
   the identical released files.** Our frozen-backbone version of CSA-Net's own mechanism (0.9349 / 70.5%) beats
   CSA-Net's published (0.91 / 63.7%); our frozen-backbone version of OutfitTransformer's mechanism (0.9349 / 74.6%)
   beats OutfitTransformer's published (0.93 / 67.1%). This is the like-for-like comparison against published work
   the cross-benchmark Recall comparison could not deliver, and it is the direct evidence for the substitutability
   claim in the framing direction: a strong frozen general-purpose representation with a light head substitutes for
   adapting a weaker backbone to the task. It remains bounded exactly as the framing says: SigLIP is far stronger
   than ResNet-18, and SigLIP fine-tuning was never tested.

2. **On the retrieval-style protocols (CIR Recall@K, FITB) ours and OutfitTransformer are comparable; on
   compatibility-AUC they are not.** FITB behaves like the CIR benchmark: ours and OT within ~1 point (ours
   significantly but slightly ahead at the ensemble level, tied at the single-model level), both well above CSA-Net.
   AUC does not: ours is ahead of both baselines by ~0.031, and the two baselines are identical to each other. **This
   narrows Claim 1 to "comparable on retrieval-style protocols"; the compatibility-AUC protocol separates the
   mechanisms.** The paper must say so.

   A hypothesis consistent with the pattern -- stated as a hypothesis, untested: the compatibility-AUC negatives are
   *random items assembled across outfits and categories*; FITB candidates and CIR pools are *same-category*. Ours was
   trained with catalog-wide (category-unrestricted) MNRL negatives; both baselines were trained with same-category
   random negatives (phase 14b / phase 34). The system whose training negative distribution matches the AUC task's
   negative distribution is the one that wins AUC, and the two systems trained on same-category negatives land at the
   same AUC. If true, this is itself an instance of Claim 3 (train/eval distribution alignment), not of an
   architectural advantage. The test would be retraining the OutfitTransformer mechanism with catalog-wide
   negatives -- out of scope here, named for the paper's limitations.

3. **Protocol sensitivity.** For ours the primary and secondary protocols agree to within 0.001 AUC / 0.04 FITB pts.
   For OutfitTransformer the primary (set-encoder query path) beats the secondary (item-alone pairwise) on FITB by
   1.5-2.4 pts and is slightly lower on AUC -- the aggregation choice matters ~1-2 points for a set encoder and not
   at all for a pooling model, which is why the choice had to be pre-declared. Untrained image+text concatenation
   (0.6420) is *worse* than untrained image alone (0.6870): text only helps once a head is trained on it.

### Step 4 -- measured inference cost (`04_inference_cost.py`; tables in `inference_cost.md`)

As built (torch parts on MPS, the rest NumPy/CPU, exactly as each phase's own evaluator), 3 full passes, median:

| System | members | precompute (s) | query + score (s) | total (s) | per member (s) | per query (ms) |
|---|---|---|---|---|---|---|
| `ours_ens` | 10 | 10.6 | 5.4 | **16.0** | 1.6 | 0.18 |
| `ot_ens` | 3 | 4.5 | 9.7 | **14.2** | 4.7 | 0.33 |
| `csa_ens` | 3 | 0.0 (+0.4 in-loop) | 26.9 | **27.0** | 9.0 | 0.91 |
| `ours_solo` | 1 | 1.2 | 0.9 | 2.3 | 2.3 | 0.03 |
| `ot_solo` | 1 | 1.5 | 3.4 | 4.9 | 4.9 | 0.11 |

**Reading.** The framing direction conceded that ensembling is "the one place our approach is genuinely more
expensive" because ours uses 10 seeds to OutfitTransformer's 3. Measured, that concession does not hold as stated:
the full 10-member ensemble of ours costs about the same wall-clock as the 3-member OutfitTransformer ensemble
(16.0 s vs 14.2 s for the entire 29,681-query benchmark), and on the query side -- the part that scales with
traffic in deployment -- ours is cheaper (5.4 s vs 9.7 s; 0.18 vs 0.33 ms per query) despite scoring 10 members
instead of 3. The reason is structural, as predicted: our query is a mean-pool over already-projected vectors,
while the set encoder needs a transformer forward per query per member. Where ours pays is precompute (10
whole-catalog projections, 10.6 s), a one-off cost. CSA-Net is the most expensive at query time (0.91 ms/query)
because every context item is embedded under a category-pair condition and compared to a per-pool candidate
tensor. Per member, ours is roughly 3x cheaper than OutfitTransformer and 5-6x cheaper than CSA-Net.

The honest statement for the paper: *ours needs more ensemble members but each is much cheaper, so end-to-end
inference cost is comparable to the transformer's 3-seed ensemble and lower on the query side; measured on one
machine, with the device split as each pipeline was actually built.* The CPU-only, device-controlled pass is
recorded below once complete.

Recorded training cost (each phase's own JSON; hardware differs by row): ours 10 seeds x ~625 s = 104 min on a
Modal cloud GPU; OutfitTransformer 3 seeds x ~2,400 s = 120 min on a Modal cloud GPU; CSA-Net 3 seeds
(2,231 + 3,560 + 2,517 s) = 138 min on the laptop's MPS. Within the two cloud rows, our full 10-seed ensemble
trained in less total GPU time than the transformer's 3-seed ensemble. The instance class was not recorded as
identical across phases, so this is indicative, not a controlled comparison. All three parity systems were
*evaluated* on the laptop; CSA-Net's parity training also ran entirely on the laptop.

### Step 5 -- parameter-span figure (`05_parameter_span_figure.py`; `figures/parameter_span.{png,pdf}`, `data/parameter_span.csv`)

Trainable parameters per member (log x: 100,485 / 998,144 / 1,705,088) against test Recall@K with 95% CIs from
step 2, filled = final ensemble, hollow = single model, untrained SigLIP as a dashed reference. The picture the
paper needs: a nearly flat band across a 17x parameter range, far above the untrained line, with the ensemble
lift visible per mechanism and ours the largest head of the three. Built and validated per the dataviz skill
(three-hue categorical palette passes all checks in light mode; the aqua slot carries a contrast warning, so
every point is direct-labelled and marker shape is a secondary encoding). Caption material: the CIs are
narrower than the marker size at this scale (about +/-0.0045 at K=10), which is itself the point of step 2.

### Parameter-ratio verification (`parameter_ratio_verification.md`)

From the papers: OutfitTransformer fine-tunes ResNet-18 end-to-end and only the fc layer of a frozen
SentenceBERT (Sec. 3.3), plus a six-layer/16-head transformer of unstated feed-forward width; CSA-Net fine-tunes
ResNet-18 end-to-end, no text. Neither states a parameter count. Estimated trainable parameters: OutfitTransformer
~12.5-14.8 M, CSA-Net ~11.2 M, ours 1.71 M -- **about 7-9x and ~7x fewer, one order of magnitude, not two.** The
framing direction's sentence has been corrected accordingly (see "Framing-direction edits" below).

### Fidelity taxonomy (`fidelity_taxonomy.md`)

Per-component, per-architecture classification with every number sourced. Two facts the paper must carry: text
is a *fidelity repair* for OutfitTransformer (the published method has it) but an *enhancement* for CSA-Net (the
published method is image-only), so CSA-Net's matched number is more generous to CSA-Net than a faithful
reproduction; and negative sampling is the one component that is neither a repair nor our defect -- phase 34's
clean single-variable +49.7% (val, single seed) / +34.2% (test, ensemble) is the number to headline, never phase
14b's 11.5x from a run whose margin never resolved.

### Step 4, continued -- device-controlled pass (every system forced onto CPU, 3 passes, median)

| System | members | precompute (s) | query + score (s) | total (s) | per member (s) | per query (ms) |
|---|---|---|---|---|---|---|
| `ours_ens` | 10 | 11.6 | 5.2 | **16.9** | 1.7 | 0.18 |
| `ot_ens` | 3 | 11.0 | 21.7 | **33.0** | 11.0 | 0.73 |
| `csa_ens` | 3 | 0.0 | 38.5 | **38.5** | 12.8 | 1.30 |
| `ours_solo` | 1 | 1.3 | 0.9 | 2.2 | 2.2 | 0.03 |
| `ot_solo` | 1 | 3.6 | 7.2 | 10.7 | 10.7 | 0.24 |

With the device held constant the picture sharpens rather than softens: **the 10-member pooling ensemble costs
half the 3-member transformer ensemble end-to-end (16.9 s vs 33.0 s) and a quarter on the query side (5.2 s vs
21.7 s; 0.18 vs 0.73 ms per query).** Per member, ours is 6-7x cheaper than either baseline. The MPS run had
flattered the transformer (its per-query forward parallelizes well on the GPU; our NumPy pooling does not use it at
all), which is why the as-built and device-controlled tables are both reported. The framing direction's concession
should be replaced with: *ours uses more ensemble members, but end-to-end inference is comparable to or cheaper than
the 3-seed transformer ensemble depending on device, and cheaper on the query side on both.*

## Overall reading -- what this phase changes for the paper

1. **Claim 1 is now statistically supported where it was an eyeball claim, and narrowed where the evidence says
   so.** Equivalent at K=10 and K=30 under a pre-declared 2% bound; a small, real edge for ours at K=50 (+1.5%,
   CI roughly +0.4% to +2.6%) that the bound cannot absorb; both mechanisms 12-14% above CSA-Net. Single models are
   not equivalent -- the transformer is ahead by 3-8% and ensembling closes it. The draft must say all of this,
   not "comparable at every K."
2. **The released-split evaluation delivers the comparison against published work that the cross-benchmark Recall
   comparison could not, and it favours the frozen-backbone systems.** All three matched-condition systems exceed all
   three published fine-tuned-backbone numbers on the identical released files. This is the like-for-like evidence
   for "task-specific backbone adaptation is substitutable by a stronger general-purpose frozen representation,"
   bounded as the framing direction already requires (SigLIP >> ResNet-18; SigLIP fine-tuning untested).
3. **The same evaluation narrows Claim 1 to retrieval-style protocols.** On FITB ours and OutfitTransformer are
   within a point (ours slightly, significantly ahead at ensemble level; tied single-model); on compatibility-AUC
   ours is ahead by 0.031 and the two baselines are identical to each other. The consistent-with-the-pattern
   hypothesis -- training-negative distribution matching the AUC task's cross-category negatives -- is an instance of
   Claim 3, stated as a hypothesis with the test that would settle it named (retrain the transformer with
   catalog-wide negatives; out of scope).
4. **Two assertions in the framing direction were wrong and are corrected.** The parameter ratio is one order of
   magnitude (~7-9x), not two. Ensembling is not where ours is more expensive: end-to-end inference is comparable
   (MPS) or half (CPU) of the transformer's 3-seed ensemble, and cheaper on the query side on both devices.
5. **Nothing in this phase changes any prior citation.** Phase 28's ensemble remains the project's best CIR result,
   phase 32's the OutfitTransformer-mechanism citation, phase 34's the CSA-Net-mechanism citation. Phase 36 adds
   intervals, a second protocol, cost, and verification to numbers that already existed; it produced no new model.

## Framing-direction edits made in this phase (approved in the plan)

- Parameter-ratio paragraph replaced with the verified one-order-of-magnitude statement and a pointer to
  `parameter_ratio_verification.md`.
- External-validity paragraph: the published AUC/FITB rows are now recorded and the work is marked done, with a
  pointer to this phase for the outcome including the Claim-1 narrowing.
- Subsequently (same day, at the user's request): Claim 1's heading, finding sentence and equivalence paragraph
  rewritten to the measured outcome (equivalent at K=10/30, small real edge at K=50, scoped to retrieval-style
  protocols with the AUC divergence stated); the ensembling-cost concession under "Efficiency and capacity"
  replaced with the measured result and explicitly withdrawn; the two related "must not be claimed" bullets
  updated; the "Outstanding work" section converted to a completed-work record pointing at this phase. The
  pre-edit backup of the framing direction was deleted at the user's request.
- Verification pass (same day, at the user's request, before drafting begins): the whole framing direction was
  re-read end to end and twelve further precision edits applied -- the external-validity section rewritten from
  future tense to a completed record (its stated aggregation had also contradicted the pre-declared CIR-native
  protocol; corrected); correction two updated from "bias direction unknown" to the phase 36 finding (equal pool
  cap, coarser granularity, plausibly easier); the header's "does not restate numbers" softened to match its
  contents; the compatibility-AUC divergence added under Claim 3 as a labelled candidate instance; the AUC gap
  re-described as a clear lead rather than "small"; the tone section amended so the introduction does not claim
  the advantage disappeared on every protocol; and pointers/corroboration sentences added under the interpretive
  contribution, the sharper reframing, the benchmark-reconstruction disclosure, and the unequal-ensembles
  disclosure. A stale-phrase sweep afterwards found only the intended historical reference to the retracted
  "two orders of magnitude" claim. The document is now consistent with every phase 36 result file.

## Disclosures this phase adds to the paper's protocol section

- **Pool cap and granularity.** Our CIR test pools are capped at 3,000 candidates per category -- the same cap
  CSA-Net's published benchmark uses. The remaining difference is granularity: our 11 coarse categories vs. their
  27 fine-grained (of 153). At matched pool size, coarse-category pools contain more heterogeneous distractors,
  which plausibly makes ours *easier*; the direction of the cross-benchmark bias is therefore not known to be
  conservative, which is one more reason that comparison cannot lead.
- **Same released test outfits under both protocols.** The official AUC/FITB files and the CIR benchmark draw on the
  same 10,000 `test.json` outfits (CIR uses 9,786). Item-level train/test overlap (34.8%) applies identically to both
  protocols and to all compared systems (phase 22).

## Files

Scripts: `scripts/common.py`, `01_capture_per_query_ranks.py`, `02_equivalence_test.py`, `03_official_auc_fitb.py`,
`03b_official_bootstrap.py`, `04_inference_cost.py`, `05_parameter_span_figure.py`. Results: `equivalence_test.md`,
`official_benchmark.md`, `official_bootstrap.md`, `inference_cost.md`, `parameter_ratio_verification.md`,
`fidelity_taxonomy.md`, `figures/parameter_span.{png,pdf}`, `data/*.json`, `data/*.npz`, `data/parameter_span.csv`,
`logs/*.log`. Total compute: under 15 minutes on the laptop, no cloud spend.

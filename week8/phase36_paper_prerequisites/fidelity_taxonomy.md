# Fidelity Repair vs. Optimization: Per-Component, Per-Architecture Taxonomy

Why this document exists: `paper/paper_framing_direction.md` requires that the improvement of each baseline
reproduction be broken down by *what kind* of change each step was, because presenting one aggregate improvement
figure invites the overclaim the framing correction exists to prevent (crediting repairs of our own incomplete
reproduction as evidence about the published literature). Every number below is pulled from the named source file;
nothing is recomputed here.

## Classification legend

| Code | Meaning |
|---|---|
| **FR** | Fidelity repair -- restores a component the *published* method has and our reproduction lacked |
| **OD** | Our own defect -- a component this project introduced (not in the published method) and left mis-set |
| **PR** | Protocol -- an evaluation/selection practice, not architecture or training recipe |
| **EN** | Enhancement -- adds something the published method does *not* have |
| **NF** | Negative-sampling finding -- a deliberate, measured departure from the published recipe |
| **OP** | Ordinary optimization -- learning rate, batch size, budget, ensembling |

## OutfitTransformer mechanism (phases 14b, 31, 32) -- validation Recall@10 chain, then test

Source: `week7/phase31_fair_baseline_outfittransformer/checkpoint_selection_check.md`, `tuning_log.md`,
`text_input_integration.md`, `phase31_notes.md`; `week7/phase32_.../individual_seeds.md`, `final_evaluation.md`;
`week4/phase14b_.../phase14b_notes.md`.

| Step | Change | Val R@10 | Relative | Code | Note |
|---|---|---|---|---|---|
| A0 | phase 14b checkpoint as shipped, evaluated on val | 0.0659 | -- | -- | starting point |
| A1 | Modal port, val-loss selection (fidelity check of the port) | 0.0637 | -3.3% | -- | reproduction noise |
| A2 (naive) | Recall@10 selection, patience left at 8 | 0.0169 | -73% | PR | the naive fix made it *worse* |
| A2-corrected | Recall@10 selection, patience recalibrated to 25 | 0.0786 | +23.4% over A1 | **PR** | two-part correction |
| A3-corrected | + text input (SigLIP text tower concat) | 0.0890 | +13.2% | **FR** | the published method uses text; our phase 14/14b reproduction did not |
| 3a | LR x batch-size grid (lr 2e-5->1.5e-4, bs 96->384) | 0.0757* | -- | OP | *measured at a cheaper 60/20 budget and at the inherited loss shape |
| 3b | uniformity_weight 1.0 -> 0.1 (single variable) | 0.0757 -> 0.1850 | **+144%** | **OD** | the regularizer was introduced by this project (phase 13/14 collapse fixes) and inherited untuned; the published method has no such term |
| 3b | margin 0.3 -> 0.2 | 0.1850 -> 0.1864 | +0.8% | OP | |
| 3c | full 100-epoch budget | 0.1924 | +3.2% | OP | |
| Step 5 | 3-seed score-averaged ensemble | 0.1942 (best solo) -> 0.2035 | +4.8% | OP | |
| *(earlier)* | negative sampling: mined -> random same-category | test 0.0051 -> 0.0588 | +934% (11.5x) | **NF** | phase 14b, *before* the parity work; the "before" run's margin never resolved -- a pathological baseline, never the representative effect size |

Test: single model 0.1799 / 0.3111 / 0.3844 (phase 31); 3-seed ensemble 0.1897 / 0.3246 / 0.4019 (phase 32).

**Aggregate decomposition, honestly stated.** From A1 (0.0637) to step 3c (0.1924) is a 3.0x gain on validation.
Its two largest components are a repair of our own regularizer (OD, +144% at its own step) and restoration of the
published method's text input (FR, +13.2%). Checkpoint-selection practice (PR) contributed +23.4% and would have
been *negative* if applied naively. Genuine optimization (LR/BS, margin, budget, ensembling) contributed the
remainder, each step under +5% except the LR/BS grid whose isolated effect is not cleanly measurable (different
budget, different loss shape). **None of OD, FR, or PR is a finding about the published OutfitTransformer.**

## CSA-Net mechanism (phases 13b, 33, 34) -- validation Recall@10 chain, then test

Source: `week7/phase33_investment_parity_csanet/phase33_notes.md`, `checkpoint_selection_check.md`;
`week7/phase34_csanet_random_negatives/single_seed_gate.md`, `final_evaluation.md`.

| Step | Change | Val R@10 | Relative | Code | Note |
|---|---|---|---|---|---|
| A0 | phase 13b checkpoint as shipped, evaluated on val | 0.0786 | -- | -- | |
| A2 | Recall@10 selection, patience unchanged at 5 | 0.0779 | -0.9% | PR | no recalibration needed -- CSA-Net's R@10 curve is near-monotone, unlike OutfitTransformer's |
| A3 | + text input | 0.0986 | +26.6% | **EN** | **the published CSA-Net is image-only** ("our method does not use text feature"); this adds a signal the original lacks |
| Step 3 | LR x batch-size tuning (winner lr=1e-4, bs=48) | 0.1075 | +9.0% | OP | |
| Step 4 | 3-seed ensemble | 0.1318 | +22.7% | OP | larger per-seed gain than OT's +4.8% |
| Phase 34 | negative sampling: mined -> random same-category, everything else byte-identical | 0.1075 -> 0.1610 (single seed) | **+49.7%** | **NF** | the clean single-variable measurement on an otherwise fully functioning parity system; the published CSA-Net uses semi-hard mining, so this is a deliberate departure |

Test: phase 33 (mined) 0.1247 / 0.2164 / 0.2748; phase 34 (random) **0.1674 / 0.2860 / 0.3586** (+34.2% / +32.2% / +30.5%).

## Ours (phases 9 -> 28) -- for symmetry; test Recall@10 unless stated

| Step | Change | Test R@10 | Relative | Code |
|---|---|---|---|---|
| phase 9 | original projection head | 0.1317 | -- | -- |
| phase 23 | Recall@10 checkpoint selection (was val-loss, best at epoch 0) + bs/tau/wd tuning | 0.1473 | +11.8% | PR + OP (selection was the largest part) |
| phase 25 | width 256 -> 1024 at re-tuned LR; depth and embed-dim rejected | 0.1505 | +2.2% | OP |
| phase 26 | 10-seed ensemble | 0.1767 | +17.4% | OP |
| phase 27 | + text input (single model, vs 0.1505) | 0.1656 | +10.0% | EN* |
| phase 28 | text + 10-seed ensemble | **0.1904** | -- | OP |

*Text is an enhancement relative to phase 9's own image-only design; there is no "published method" for ours to be
faithful to, so FR/OD do not apply.

## Consequences for the paper

1. **Text cannot be one row in any component table.** For OutfitTransformer it restores fidelity (FR); for CSA-Net it
   goes beyond the published method (EN). Consequence: CSA-Net's matched-condition number is *more generous to
   CSA-Net* than a faithful reproduction would be, and OutfitTransformer's matched-condition number is *closer to
   faithful* than its un-invested one was. Both directions must be visible.
2. **The regularizer correction is not evidence about anyone's published work.** It is the largest single lever in the
   OutfitTransformer chain and it is a defect this project introduced. Quote it as such.
3. **Negative sampling is the one component that survives the framing correction intact.** Phase 34 changed exactly
   one variable on a functioning matched system and measured +49.7% (val, single seed) / +34.2% (test, ensemble). It
   is neither a fidelity repair nor our defect, which is why Claim 2 holds top-level position. Headline phase 34's
   number, never phase 14b's 11.5x (pathological baseline; cite it as failure-mode severity only).
4. **The checkpoint-selection correction is two-part**, and the two architectures needed different amounts of it
   (OutfitTransformer: patience 8 -> 25 was essential; CSA-Net: none). A paper recommending "select on the retrieval
   metric" without "and re-tune patience for its noise" is giving advice that hurt one of the two architectures.
5. **Both baselines' investment histories are asymmetric with each other and with ours**; the honest summary sentence
   is that reproduction fidelity, protocol, a self-inflicted regularizer defect, an enhancement, one genuine
   negative-sampling finding, and ordinary optimization *all* moved the reported baseline numbers substantially -- and
   only the last two say anything about the methods themselves.

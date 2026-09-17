# Phase 31, Step 3: Hyperparameter Tuning

Base configuration going into this step: A3-corrected (`input_mode="image_text"`, `selection_metric=
"recall10"`, patience=25, phase 14b's original `lr=2e-5, batch_size=96, margin=0.3, uniformity_weight=1.0`)
-- val Recall@10 = 0.0890 (see `checkpoint_selection_check.md`).

Sweep budget for steps 3a/3b: `max_epochs=60, patience=20` -- NOT phase 23's original 40/8. Per
`checkpoint_selection_check.md`'s own diagnosis, Recall@10 has a genuine ~17-epoch noisy plateau early in
training on this architecture; patience=8 at a 40-epoch cap would very likely re-trigger the same
premature stop across most of this grid, making the comparison meaningless. patience=20/max_epochs=60 is a
deliberately cheaper proxy for RANKING configs relatively, reconciled against the full budget in step 3c.

## Step 3a: joint LR x batch-size grid (15 configs + 1 bracket extension)

`LR_GRID = [1e-5, 2e-5, 5e-5, 1.5e-4, 5e-4]` x `BS_GRID = [96, 192, 384]`, centered wider/upward from the
current lr=2e-5 rather than symmetrically (per `architecture_notes.md`: phase 14b's 92-epoch run never
resolved its triplet margin at any epoch -- under-training, not over-stepping, is the diagnosed failure
mode, and 2e-5 was inherited from a reference repo that fine-tunes a full backbone, not a small
frozen-backbone head).

| LR | Batch size | Val R@10 | Best epoch | Epochs run |
|---|---|---|---|---|
| **1.5e-4** | **384** | **0.0757** | 54 | 60 |
| 5e-5 | 96 | 0.0727 | 52 | 60 |
| 5e-5 | 192 | 0.0724 | 53 | 60 |
| 1.5e-4 | 192 | 0.0715 | 47 | 60 |
| 5e-4 | 384 | 0.0655 | 54 | 60 |
| 2e-5 | 96 | 0.0529 | 49 | 60 |
| 5e-5 | 384 | 0.0523 | 52 | 60 |
| 5e-4 | 192 | 0.0434 | 9 | 30 |
| 1.5e-4 | 96 | 0.0424 | 13 | 34 |
| 2e-5 | 192 | 0.0392 | 51 | 60 |
| 5e-4 | 96 | 0.0347 | 6 | 27 |
| 1e-5 | 96 | 0.0306 | 49 | 60 |
| 2e-5 | 384 | 0.0255 | 48 | 60 |
| 1e-5 | 192 | 0.0242 | 48 | 60 |
| 1e-5 | 384 | 0.0153 | 11 | 32 |

**Winner: lr=1.5e-4, bs=384 (val R@10=0.0757)**, sitting at BS_GRID's top edge -- per the bracketing rule
(phase 23's own tau-sweep lesson: don't leave an unbracketed optimum), extended with bs=768:

| LR | Batch size | Val R@10 |
|---|---|---|
| 1.5e-4 | 768 | 0.0701 (worse -- bracket resolved, 384 confirmed) |

*Naming note*: the auto-generated config names show `lr1e-4` for the 1.5e-4 rows -- a cosmetic artifact
of Python's `f"{1.5e-4:.0e}"` formatting rounding the mantissa for display, NOT a training error. Every
number in this table was cross-checked directly against each config's actual stored `lr` field.

## Step 3b: loss-shape sweep (extension beyond the brief's literal ask, clearly labelled)

The brief's step 3 names only LR and batch size. Added anyway, labelled explicitly as an extension:
(i) phase 23's own discipline, which the brief invokes by name, included an analogous loss-shaping sweep
(temperature) after its LR/BS grid; (ii) `checkpoint_selection_check.md`'s own finding showed the
uniformity regularizer dominated phase 14b's original (broken) val_loss selection signal -- leaving
`uniformity_weight` at its inherited value of 1.0 untouched would be an odd place to stop investing, in a
phase whose entire purpose is removing under-investment.

At the step 3a winner (lr=1.5e-4, bs=384):

**Uniformity weight sweep, coarse then refined around the peak:**

| uniformity_weight | Val R@10 |
|---|---|
| 0.0 | 0.0111 (collapsed -- confirms the uniformity term is genuinely needed, just not at weight 1.0) |
| **0.1** | **0.1850** |
| 0.15 | 0.1829 |
| 0.2 | 0.1788 |
| 0.25 | 0.1758 |
| 0.3 | 0.1699 |
| 0.5 | 0.1375 |
| 1.0 (phase 14b's original) | 0.0757 |

**This is the single largest lever found in this entire tuning step.** Reducing `uniformity_weight` from
phase 14b's inherited 1.0 to 0.1 alone takes val Recall@10 from 0.0757 to 0.1850 -- a **+144% relative**
gain, holding LR/batch size fixed. Directly mechanistically connected to `checkpoint_selection_check.md`'s
own finding: phase 14b's triplet margin was never satisfied across its entire 92-epoch run, while
`val_loss` (dominated by the uniformity term at weight 1.0) kept "improving" regardless -- this sweep shows
that same over-weighted uniformity term wasn't just corrupting the selection *signal*, it was actively
suppressing the *training* itself. A sharp cliff exists between 0.0 (collapse) and 0.1 (best found), not
explored further below 0.1 -- diminishing-returns territory not worth chasing past this point given the
scale of the gain already captured.

**Margin sweep at uniformity_weight=0.1:**

| margin | Val R@10 |
|---|---|
| 0.1 | 0.1834 |
| **0.2** | **0.1864** |
| 0.3 (phase 14b's original) | 0.1850 |
| 0.5 | 0.1800 |

A modest, real gain (margin=0.2 beats margin=0.3 by +0.8% relative) -- small compared to the
uniformity-weight lever, but free.

## Step 3c: budget check

Step 3a/3b's winner (lr=1.5e-4, bs=384, uniformity_weight=0.1, margin=0.2) consistently hit its best epoch
in the mid-50s across the sweep -- close to the 60-epoch cap, suggesting the reduced sweep budget may have
cut off real, still-in-progress improvement. Re-run at the full budget that worked cleanly for
A2/A3-corrected (max_epochs=100, patience=25):

| Budget | Val R@10 | Best epoch |
|---|---|---|
| Sweep (60 epochs, patience=20) | 0.1864 | 54 |
| **Full (100 epochs, patience=25)** | **0.1924** | 83 |

+3.2% relative -- confirms the sweep budget was genuinely truncating some real improvement, but only
modestly; the sweep's relative rankings are validated as a reasonably reliable (if slightly conservative)
proxy. **0.1924 is step 3's final, adopted number**, and `max_epochs=100, patience=25` becomes the
standing training budget for steps 4 and 5.

## Step 3 summary: final winning configuration

`input_mode="image_text", selection_metric="recall10", lr=1.5e-4, batch_size=384, uniformity_weight=0.1,
margin=0.2, max_epochs=100, patience=25` -- **val Recall@10 = 0.1924**.

| Stage | Val R@10 | Relative gain vs. prior stage |
|---|---|---|
| A0 (phase 14b as-shipped) | 0.0659 | -- |
| A3-corrected (selection fix + text) | 0.0890 | +35.1% |
| Step 3a (+ LR/BS tuning) | 0.0757* | see note |
| Step 3b (+ loss-shape tuning) | 0.1864 | +146.4% over 3a |
| Step 3c (+ full-budget training) | **0.1924** | +3.2% over 3b |

*Step 3a's own number (0.0757) is measured at phase 14b's original loss-shape hyperparameters
(uniformity_weight=1.0, margin=0.3) -- LOWER than A3-corrected's 0.0890, because 3a used the cheaper
60-epoch/patience=20 sweep budget rather than A3-corrected's full 100-epoch/patience=25 run. LR/BS tuning's
own isolated contribution can't be cleanly read off this row alone; what matters is the final, fully-tuned
number (0.1924), a **+116.2% relative gain over A3-corrected** from the combination of LR/batch-size and
loss-shape tuning together, the large majority of which traces to the uniformity_weight correction.

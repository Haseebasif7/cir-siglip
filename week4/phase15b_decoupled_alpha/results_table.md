# Phase 15b: Fully Decoupled Alpha Conditioning -- Results Table

Evaluated on this project's own CIR harness
(`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged since
phase 12): 29,681 queries, 0 skipped. Full 11-point alpha sweep in
`data/cir_sweep_decoupled.json`.

## Full alpha sweep, this phase's decoupled-trained checkpoint

| alpha | Recall@10 | Recall@30 | Recall@50 | axis1 (visual sim) | axis2 (hit rate) | overlap-raw | overlap-alpha0 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0673 | 0.1331 | 0.1792 | 0.7037 | 0.0720 | 0.0988 | 1.0000 |
| 0.1 | 0.0673 | 0.1331 | 0.1792 | 0.7037 | 0.0720 | 0.0988 | 1.0000 |
| 0.2 | 0.0673 | 0.1331 | 0.1792 | 0.7037 | 0.0720 | 0.0988 | 0.9996 |
| 0.3 | 0.0675 | 0.1331 | 0.1792 | 0.7037 | 0.0720 | 0.0990 | 0.9966 |
| 0.4 | 0.0678 | 0.1333 | 0.1796 | 0.7037 | 0.0720 | 0.0984 | 0.9808 |
| 0.5 | 0.0683 | 0.1355 | 0.1815 | 0.7040 | 0.0690 | 0.1016 | 0.8992 |
| 0.6 | 0.0684 | 0.1378 | 0.1865 | 0.7015 | 0.0640 | 0.1046 | 0.6482 |
| 0.7 | 0.0590 | 0.1260 | 0.1737 | 0.6918 | 0.0550 | 0.0916 | 0.3598 |
| 0.8 | 0.0484 | 0.1104 | 0.1547 | 0.6821 | 0.0450 | 0.0830 | 0.2268 |
| 0.9 | 0.0440 | 0.1030 | 0.1452 | 0.6780 | 0.0450 | 0.0768 | 0.1918 |
| 1.0 | 0.0434 | 0.1011 | 0.1434 | 0.6773 | 0.0420 | 0.0752 | 0.1848 |

**Shape note**: this is NOT a smooth linear dial -- Recall@K and every
diagnostic stay essentially flat (even slightly RISING) from alpha=0.0 to
~0.5-0.6, then drop sharply from ~0.6 to 1.0. This matches the
attention-weight-shift probe's own finding (`training_log.md`): the
underlying attention weights are now saturated near-binary switches (mean
L1 shift 1.9955, essentially the theoretical max of 2.0) rather than
gradually interpolating -- so retrieval behavior stays near the alpha=0
regime until the softmax's own nonlinearity pushes the mixture over a
threshold, then transitions quickly. The overlap-with-raw diagnostic even
moves the WRONG direction late in the sweep (0.099 at alpha=0 -> peak 0.105
at alpha=0.6 -> 0.075 at alpha=1.0, i.e. LOWER substitute-mode overlap with
raw SigLIP than the complement endpoint has) -- opposite of every other
mechanism in this project (where higher alpha = more raw-SigLIP-like), a
genuinely unexpected finding flagged here rather than smoothed over.

## Step 2's pre-declared stop condition, applied

**Bar** (declared before training): realized Recall@10 range must close at
least half the gap between phase 15 discrete's range (~3.5% relative) and
phase 12c's range (~31% relative) -- i.e. reach roughly 15-17% relative or
more -- **without either endpoint's Recall@K dropping meaningfully below
phase 15's discrete-trained endpoints.**

### Part 1: the range bar

| | Recall@10 range (alpha 0->1, relative) |
|---|---|
| Phase 15 discrete (prior best) | 3.5% |
| Phase 12c/12d (target reference) | 31% |
| **This phase (decoupled)** | **35.6%** |

**Clears the range bar decisively** -- 35.6% relative range, more than
double the 15-17% threshold, and even exceeds phase 12c/12d's own reference
range (31%). (R@30 range: 24.0%; R@50 range: 20.0% -- both also comfortably
clear the bar.)

### Part 2: the no-endpoint-regression constraint

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 15 discrete, alpha=0 | 0.0657 | 0.1326 | 0.1792 |
| **This phase, alpha=0** | **0.0673** | **0.1331** | **0.1792** |
| Ratio | 102.5% | 100.4% | 100.0% |
| Phase 15 discrete, alpha=1 | 0.0634 | 0.1294 | 0.1754 |
| **This phase, alpha=1** | **0.0434** | **0.1011** | **0.1434** |
| Ratio | **68.4%** | **78.1%** | **81.8%** |

**The complement endpoint (alpha=0) does NOT regress** -- it matches or
very slightly exceeds phase 15 discrete's own complement endpoint at every
K. **But the substitute endpoint (alpha=1) drops substantially**: to 68.4%
of phase 15 discrete's own alpha=1 Recall@10 (a 31.6% relative drop), 78.1%
at R@30 (21.9% drop), 81.8% at R@50 (18.2% drop). This is a real, one-sided,
consistent-in-direction regression across all three K -- not noise -- and
exceeds this project's own established "meaningful" threshold (the ±20%
tolerance band used for endpoint-consistency checks throughout phases
13b/15) at R@10 and R@30, and sits right at that boundary at R@50.

### Verdict on the compound condition

The bar is a conjunction (range AND no endpoint regression), not either
condition alone. **Part 1 is cleared decisively. Part 2 is NOT cleared** --
the substitute endpoint regressed meaningfully. Per the brief's own
pre-declared rule, **the compound stop condition is NOT met.** See
`phase15b_notes.md` for the full verdict and what this means for the
architectural thread going forward.

## Against this project's own configurations and both published baselines

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 14: OutfitTransformer mechanism | 0.0201 | 0.0588 | 0.0911 |
| This phase, alpha=1.0 (substitute-leaning) | 0.0434 | 0.1011 | 0.1434 |
| Raw SigLIP (alone, no training) | 0.0553 | 0.1067 | 0.1437 |
| Phase 15, continuous, alpha=1 | 0.0596 | 0.1236 | 0.1693 |
| Phase 15, discrete, alpha=1 | 0.0634 | 0.1294 | 0.1754 |
| Phase 12c: Substitute mode | 0.0667 | 0.1307 | 0.1734 |
| Phase 15, discrete, alpha=0 | 0.0657 | 0.1326 | 0.1792 |
| **This phase, alpha=0.0 (complement-leaning)** | **0.0673** | **0.1331** | **0.1792** |
| **This phase, alpha=0.6 (peak)** | **0.0684** | **0.1378** | **0.1865** |
| Phase 13b: CSA-Net mechanism (fixed, non-conditional) | 0.0725 | 0.1393 | 0.1844 |
| CSA-Net published | 0.0827 | 0.1567 | 0.2091 |
| OutfitTransformer published | 0.0958 | 0.1796 | 0.2198 |
| Phase 12c: Blend (0.5) | 0.0893 | 0.1695 | 0.2255 |
| Phase 12c: Complement mode | 0.0971 | 0.1875 | 0.2471 |
| Phase 9 Model A (alone) | 0.1317 | 0.2464 | 0.3216 |

(Ordered by Recall@10, ascending.) This phase's complement-leaning end (and
its peak at alpha=0.6) is this project's best-ever result for the
CSA-Net-plus-conditioning architectural thread, essentially matching phase
13b's fixed non-conditional CSA-Net at R@50 (97.2%) and coming close at
R@10/30 (92.8%/95.7%). Its substitute-leaning end, however, is now the
WORST-performing substitute-style configuration this project has evaluated
on this harness except for phase 14's OutfitTransformer and raw SigLIP.

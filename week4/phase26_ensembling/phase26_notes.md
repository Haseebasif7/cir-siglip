# Phase 26: Ensembling -- The Biggest Single Gain in This Whole Tuning/Scale/Ensemble Sequence

## The headline number

Score-averaging ten independently-seeded copies of phase 25's winning architecture (hidden_dims=[1024], out_dim=128, lr=0.0005) produced test-benchmark Recall@10/30/50 = **0.1767/0.3054/0.3828**, up from phase 25's single-model 0.1505/0.2740/0.3503 -- **+17.4%/+11.5%/+9.3% relative**. This is a substantially bigger jump than either phase 23/24's hyperparameter tuning (+11.8%/+8.9%/+7.0% relative over phase 9) or phase 25's architecture scaling (+2.2%/+2.1%/+1.8% relative over phase 23/24). Ensembling is, by a clear margin, the most effective lever pulled anywhere in this sequence so far.

## Why this is surprising given how tight the individual seeds were

`individual_seeds.md` shows the ten solo models landing in a narrow band -- validation Recall@10 from 0.1617 to 0.1660, spread of only 0.0043, std 0.0014. By the brief's own stated logic ("if individual seeds vary a lot from each other, that's a sign ensembling has real room to help... if they're all nearly identical, that sets expectations that ensembling's benefit may be modest"), this tight a spread would suggest a modest ensembling payoff. That is not what happened: even a 2-member ensemble (seeds 42+1, val Recall@10=0.1767) already beats every individual solo seed by a wide margin (best solo was 0.1660), and the gain keeps compounding as more members are added. The lesson: near-identical *peak* scores do not mean near-identical *error patterns*. Ten runs converging to similar Recall@10 numbers can still be right and wrong on different individual queries -- disagreement that only shows up when you look at which specific items each model ranks well, not at their aggregate score. Score-averaging exploits exactly that per-query disagreement, and it turned out to be substantial here despite the tight solo-score band.

## The ensemble size sweep: real diminishing returns, but not fully flat by 10

| Size | val Recall@10 | Gain over solo (0.1656) |
|---|---|---|
| 1 (solo) | 0.1656 | -- |
| 2 | 0.1767 | +0.0111 |
| 3 | 0.1805 | +0.0149 |
| 5 | 0.1854 | +0.0198 |
| 7 | 0.1852 | +0.0196 (essentially flat vs size 5) |
| 10 | 0.1869 | +0.0213 |

Most of the total gain arrives immediately -- going from 1 to 2 members alone captures more than half of the eventual 10-member improvement. Growth from 2 onward is real but shrinking (2->3: +0.0038, 3->5: +0.0049 over two more members, 5->7: essentially nothing, 7->10: +0.0017 over three more members). This is the classic ensemble diminishing-returns curve, not a straight line -- size=10 is still the best point tested, but the marginal member is buying very little by the end. Worth naming for any future session considering more than 10 members: the curve strongly suggests it would keep improving only slightly, not proportionally, for members 11-20+.

## Does architectural diversity help beyond seed diversity? No, not here

`architectural_diversity_check.md`: swapping 2 of the 10 width=1024 members for 2 freshly-trained width=512 models (phase 25's own winning width=512 setting, lr=0.0005) at the same total ensemble size (10) scored *slightly worse* (0.1865 vs 0.1869) than the same-architecture-only ensemble. The reason is simple and visible in the numbers: the width=512 solo models (0.1610, 0.1624) are individually weaker than every width=1024 solo model (0.1617-0.1660) -- consistent with phase 25's own width-sweep finding that width=1024 beats width=512 at the correct learning rate. Mixing in a weaker architecture cost slightly more in per-member quality than it bought in diversity. This is a useful negative result, not a wash: it says the gain observed here is a **seed-diversity effect specifically**, not a general "any kind of diversity helps" effect -- diluting the ensemble with a systematically weaker architecture doesn't pay for itself. It's worth noting separately that the 2-member width=512-only ensemble (0.1733) also beats both of its own solo members (0.1610, 0.1624) by a wide margin, exactly mirroring the width=1024 pattern -- so the underlying score-averaging mechanism itself generalizes across architectures, it's specifically *mixing* unequal-strength architectures into one ensemble that didn't help here.

## Test-benchmark result exceeded the validation-benchmark gain

The validation benchmark showed +12.9% relative Recall@10 (0.1656 solo -> 0.1869 ensemble at size 10). The test benchmark showed an even larger +17.4% relative gain (0.1505 -> 0.1767). This is good news, not a red flag: the test benchmark is much larger (29,681 queries vs 22,595 validation queries), the selection decision (best composition = all 10 same-architecture members) was made purely on the validation benchmark before the test benchmark was ever touched, and the gain direction and rough magnitude carried through cleanly rather than evaporating -- exactly what should happen when a technique's benefit is real rather than validation-benchmark noise.

## Final, honest verdict

**Ensembling works, and it works better than tuning or scaling did.** Ten independently-seeded copies of the same small architecture, combined by averaging their similarity scores (never their raw embeddings, since independently-initialized models share no common coordinate system), delivered the single largest relative improvement seen anywhere in this hyperparameter-tuning -> temperature-ceiling -> scale -> ensembling sequence. The gain shows clear diminishing returns past about 5 members but had not fully flattened by 10. Architectural diversity, tested as a smaller secondary check, did not add anything beyond seed diversity in this instance -- the two width=512 members were individually weaker and diluted rather than strengthened the ensemble.

## Practical cost note

Ten training runs of this architecture cost about the same wall-clock time as any single phase-25 sweep column (each run converges in ~1-6 epochs, ~650-700s), and Modal's `.map()` ran all 11 new configs (9 new width=1024 seeds + 2 width=512 diversity seeds) in parallel, so the entire phase's training cost was roughly one sweep's worth of wall-clock time, not ten times a single run's. The main added cost at *inference* time is proportional to ensemble size (10x the forward passes of a single model) -- worth flagging explicitly for any future deployment-facing phase, since this phase only evaluated retrieval quality, not serving cost.

## Final reference configuration going forward

- **Ensemble**: 10 independently-seeded copies of hidden_dims=[1024], out_dim=128 (lr=0.0005, batch_size=256, weight_decay=0.0, tau=0.15, R=8 -- all unchanged from phase 25). Seeds: 42 (phase 25's own confirmatory-retrain checkpoint, reused directly) plus 1-9 (trained fresh this phase).
- **Combination method**: average per-query-per-candidate cosine similarity scores across all 10 members, then rank by the averaged score. Never average raw embedding vectors.
- **Test-benchmark result**: Recall@10/30/50 = 0.1767/0.3054/0.3828.
- **Checkpoints**: `week4/phase26_ensembling/models/ensemble_seed{42,1..9}.pt` (10 files, ~3.7MB each).

## Operational footnote

Training used the same `caffeinate -is`-wrapped Modal `.map()` pattern established in phase 25 (after that phase's two RemoteError failures traced to the local machine sleeping mid-run); all 11 new training runs plus the checkpoint downloads completed cleanly on the first attempt this time, no retries needed.

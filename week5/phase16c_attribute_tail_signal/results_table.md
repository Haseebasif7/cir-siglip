# Phase 16c: Results Table

## Retrieval accuracy and axis checks across the alpha sweep

| alpha | Hit Rate@5 | Hit Rate@10 | Mean ref_count@5 | Mean ref_count@10 |
|---|---|---|---|---|
| 0.0 | 0.2810 | 0.3563 | 17.97 | 17.39 |
| 0.1 | 0.2810 | 0.3558 | 18.00 | 17.40 |
| 0.2 | 0.2804 | 0.3552 | 18.33 | 17.45 |
| 0.3 | 0.2810 | 0.3510 | 18.18 | 17.43 |
| 0.4 | 0.2788 | 0.3531 | 18.02 | 17.45 |
| 0.5 | 0.2804 | 0.3531 | 17.97 | 17.51 |
| 0.6 | 0.2794 | 0.3526 | 17.96 | 17.46 |
| 0.7 | 0.2783 | 0.3526 | 17.94 | 17.46 |
| 0.8 | 0.2772 | 0.3520 | 17.84 | 17.42 |
| 0.9 | 0.2778 | 0.3510 | 17.84 | 17.39 |
| 1.0 | 0.2783 | 0.3510 | 17.59 | 17.27 |

## Direct comparison to phase 16's endpoints

| | Phase 16 tail (a=0) | Phase 16 relevance (a=1) | Phase 16c tail (a=0) | Phase 16c relevance (a=1) |
|---|---|---|---|---|
| Hit Rate@5 | 0.3194 | 0.3184 | 0.2810 | 0.2783 |
| Hit Rate@10 | 0.3958 | 0.3948 | 0.3563 | 0.3510 |
| Mean ref_count@5 | 16.90 | 17.49 | 17.97 | 17.59 |
| Mean ref_count@10 | 16.58 | 17.21 | 17.39 | 17.27 |

## Axis checks (endpoints, alpha=1.0 relevance vs alpha=0.0 tail-exposure)

- **Axis 1 (relevance mode's Hit Rate@K should exceed tail mode's, every K): FAIL** -- K=5: relevance=0.2783 vs tail=0.2810, K=10: relevance=0.3510 vs tail=0.3563
- **Axis 1, CLEAR-GAP bar (this phase's own success condition -- a real, non-noise gap, not just movement in the right direction; phase 16's own gap was ~0.001, noise-level; requiring a gap > 0.02 here to count as 'clear'): NOT CLEAR** -- gap@5=-0.0027, gap@10=-0.0053
- **Axis 2 (tail mode's mean retrieved ref_count should be lower than relevance mode's, every K): FAIL** -- K=5: relevance=17.59 vs tail=17.97, K=10: relevance=17.27 vs tail=17.39

## Field-standard long-tail metrics

Same formulas and catalog-wide denominators as phase 16 (3,777,545 total items, 1,888,773 tail-tier items).

### N=10

| alpha | Coverage@N | Tail-Coverage@N | APRI | RPI |
|---|---|---|---|---|
| 0.0 | 0.2488% (9397) | 0.2021% (3817) | 17.39 | 0.5072 |
| 0.1 | 0.2487% (9396) | 0.2022% (3820) | 17.40 | 0.5072 |
| 0.2 | 0.2481% (9373) | 0.2017% (3810) | 17.45 | 0.5061 |
| 0.3 | 0.2475% (9349) | 0.2016% (3808) | 17.43 | 0.5048 |
| 0.4 | 0.2471% (9334) | 0.2012% (3801) | 17.45 | 0.5035 |
| 0.5 | 0.2467% (9320) | 0.2015% (3805) | 17.51 | 0.5019 |
| 0.6 | 0.2467% (9319) | 0.2019% (3814) | 17.46 | 0.5006 |
| 0.7 | 0.2462% (9300) | 0.2015% (3805) | 17.46 | 0.4998 |
| 0.8 | 0.2454% (9271) | 0.2010% (3796) | 17.42 | 0.4981 |
| 0.9 | 0.2451% (9259) | 0.2018% (3812) | 17.39 | 0.4973 |
| 1.0 | 0.2443% (9228) | 0.2012% (3800) | 17.27 | 0.4972 |

### N=20

| alpha | Coverage@N | Tail-Coverage@N | APRI | RPI |
|---|---|---|---|---|
| 0.0 | 0.3720% (14052) | 0.3113% (5879) | 17.19 | 0.5015 |
| 0.1 | 0.3712% (14023) | 0.3115% (5883) | 17.25 | 0.5015 |
| 0.2 | 0.3708% (14008) | 0.3115% (5883) | 17.33 | 0.5008 |
| 0.3 | 0.3703% (13990) | 0.3115% (5884) | 17.31 | 0.5002 |
| 0.4 | 0.3699% (13974) | 0.3117% (5888) | 17.31 | 0.4987 |
| 0.5 | 0.3693% (13950) | 0.3119% (5891) | 17.34 | 0.4978 |
| 0.6 | 0.3689% (13934) | 0.3130% (5911) | 17.39 | 0.4965 |
| 0.7 | 0.3683% (13911) | 0.3128% (5908) | 17.35 | 0.4955 |
| 0.8 | 0.3679% (13899) | 0.3128% (5909) | 17.34 | 0.4946 |
| 0.9 | 0.3671% (13867) | 0.3130% (5911) | 17.40 | 0.4932 |
| 1.0 | 0.3670% (13862) | 0.3135% (5921) | 17.39 | 0.4920 |

### Direct comparison to phase 16's endpoints

- N=10: **Phase 16** Coverage@N 0.2382% (tail) vs 0.2384% (relevance), APRI 16.58 vs 17.21, RPI 0.5230 vs 0.5233. **Phase 16c** Coverage@N 0.2488% (tail) vs 0.2443% (relevance), APRI 17.39 vs 17.27, RPI 0.5072 vs 0.4972.
- N=20: **Phase 16** Coverage@N 0.3584% (tail) vs 0.3583% (relevance), APRI 16.23 vs 16.92, RPI 0.5149 vs 0.5171. **Phase 16c** Coverage@N 0.3720% (tail) vs 0.3670% (relevance), APRI 17.19 vs 17.39, RPI 0.5015 vs 0.4920.

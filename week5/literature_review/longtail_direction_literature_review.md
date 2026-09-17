# Literature Review: Controllable, Item-Only Long-Tail Visual Retrieval

## Context

The project spent weeks (phases 5 through 15b, see `PROJECT_MEMORY.md` and
`week4/`) building and rigorously testing a substitute/complement
controllable retrieval mechanism, benchmarked against CSA-Net and
OutfitTransformer reproductions. That thread is not abandoned, it is
paused, with a working, verified controllable dial (phase 12c/12d) as a
real, reusable asset. It was paused because both literature baselines
(CSA-Net, OutfitTransformer) turned out to have no usable official code and
proved too narrow to build a strong comparison story around once the fair
reproductions were actually built (phases 13-15b).

This document is the literature foundation for a new direction, reached
through an actual literature scan rather than a guess: **controllable,
item-to-item, visual long-tail exposure in product recommendation.** It
reuses the same SigLIP backbone, the same proven controllable dial
mechanism, and phase 3's already-built popularity metrics (ARP, catalog
coverage), redirected at a different, currently active, better-resourced
research problem than CSA-Net/OutfitTransformer turned out to be.

Same card format as the week 1 literature review
(`week1/lit_review_output/lit_review_extraction.md`): Technique/model used,
How it's used, Dataset/category tested on, Metrics used, Key
number/result, Stated weakness/limitation. Where the source material available
for this review did not specify a field for a given paper, that field is
marked **not confirmed** rather than filled in with a plausible-sounding
guess, per this review's own instructions.

---

## 1. Evaluating the Long Tail in Recommendation System: A Systematic Review of Approaches, Datasets and Metrics

Saha, Biswas, Das & Paitya, January 2026 | International Journal of
Mathematical, Engineering and Management Sciences, DOI
10.33889/IJMEMS.2026.11.1.019

Not a technique paper, this is a PRISMA 2020 systematic review of 71 papers
(2012-2024) on long-tail recommendation. Included as the map of the field,
not as something to compare against or reproduce.

- **Technique/model used:** Not applicable, this is a meta-analysis, not a
  proposed method. Its own methodology is a PRISMA 2020 systematic
  literature review: a structured search, screening, and categorization of
  71 prior papers.
- **How it's used:** Categorizes long-tail recommendation approaches from
  the reviewed literature into six families: cluster-based, graph-based,
  deep learning/neural network-based, multi-objective, traditional, and
  other. Also consolidates the field's standard evaluation practice into a
  metrics taxonomy: accuracy-based (Recall@N, Precision@N, F1, Accuracy),
  error-based (MAE, RMSE), ranking-based (MRR, MAP, Hit Ratio, AUC, nDCG),
  and coverage/popularity-based (Coverage, Tail-Coverage, Average
  Popularity of Recommended Items/APRI, Ratio of Popular Items/RPI), plus
  diversity and novelty metrics. This project should align its own existing
  ARP/coverage code (built in phase 3) with this taxonomy's exact
  terminology and formulas rather than inventing project-specific naming.
- **Dataset/category tested on:** Not applicable in the usual sense (this
  paper doesn't run experiments), but it reports which datasets the 71
  reviewed papers used: MovieLens dominates the field at 37 of 71 papers,
  followed by Netflix, Amazon, and BookCrossing.
- **Metrics used to evaluate the system:** Not applicable, see above; the
  metrics taxonomy described is the paper's actual contribution, not a
  metric used to evaluate a proposed model.
- **Key number/result:** MovieLens alone accounts for 37 of the 71 reviewed
  papers (roughly 52%), confirming the field leans heavily on one dataset
  family. The exact formulas for APRI, RPI, and Tail-Coverage are referenced
  by name in the source material available for this review but their
  precise mathematical definitions were **not confirmed** here, follow up
  against the paper directly before using them in any metric-alignment
  code.
- **Stated weakness/limitation:** **Not confirmed** in the source material
  available for this review, flag for follow-up if the paper's own stated
  limitations need to be cited directly.

---

## 2. SAGERec: Sampling and Gating for Enhanced Long-Tail Item Recommendations

Alshabanah, Yang & Annavaram, February 2026 | WSDM 2026 | Code:
`github.com/alshabae/SAGERec`

Confirms the long-tail recommendation field is an active 2026 research
area, not a stale one, but structurally a different kind of method from
what this project builds.

- **Technique/model used:** Learns user representations by identifying the
  most informative items in a user's interaction history via a trainable
  sampler, then fuses two expert models through a tail-aware gating
  network.
- **How it's used:** The trainable sampler selects which items in a user's
  history are most informative before they're used to build that user's
  representation; the tail-aware gate then combines two expert models'
  outputs, presumably weighting toward whichever expert handles long-tail
  items better, though the exact gating mechanics were **not confirmed**
  in the source material available for this review.
- **Dataset/category tested on:** **Not confirmed** in the source material
  available for this review, flag for follow-up before citing specific
  benchmarks.
- **Metrics used to evaluate the system:** **Not confirmed** in the source
  material available for this review.
- **Key number/result:** **Not confirmed** in the source material
  available for this review.
- **Stated weakness/limitation:** Structural, noted here rather than in the
  paper itself: this is fundamentally a user-representation,
  collaborative-filtering-style method, it requires rich individual user
  interaction histories. That's a structural mismatch with this project's
  item-only direction, so it's cited as evidence the field is active, not
  treated as a direct baseline.

---

## 3. SAGE: Global Semantic Alignment with LLMs for Long-Tail Sequential Recommendation

2026 | Web Conference/WWW 2026 | Code:
`github.com/Applied-Machine-Learning-Lab/WWW2026_SAGE-LLM`

The other current 2026 paper confirming this field is live right now.

- **Technique/model used:** LLM-based semantic alignment, so that tail
  items can inherit features from semantically related head items.
- **How it's used:** Presumably aligns a tail item's representation toward
  semantically similar head items' representations using an LLM's semantic
  understanding, closing the exposure gap that comes from tail items having
  too little interaction data of their own. The exact alignment mechanism
  was **not confirmed** in the source material available for this review.
- **Dataset/category tested on:** **Not confirmed** in the source material
  available for this review.
- **Metrics used to evaluate the system:** **Not confirmed** in the source
  material available for this review.
- **Key number/result:** **Not confirmed** in the source material
  available for this review.
- **Stated weakness/limitation:** Same structural note as SAGERec above:
  this is a user-sequence-based method (it operates on sequential user
  interaction history), a structural mismatch with this project's
  item-only direction.

---

## 4. GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation

Lin, Meng, Wang, Long, Zhou & Xiao, 2024 | CIKM 2024 | arXiv 2407.12338 |
Code: `github.com/NanGongNingYi/GUME`

The closest prior work found in this review, visual features plus
long-tail recommendation, and the one paper here that needed the most
careful checking for redundancy with this project's own direction.

- **Technique/model used:** Enhances the user-item interaction graph using
  multimodal (visual plus text) similarity between items, to improve
  connectivity and representation quality specifically for long-tail
  items, combined with a separate user modality enhancement strategy.
- **How it's used:** Item-item multimodal similarity edges are added into
  the user-item graph, giving sparsely-connected long-tail items extra,
  content-derived connectivity that pure interaction data wouldn't provide
  on its own. Visual features are 4096-dimensional, extracted via an
  older-style pretrained CNN encoder, comparable to VGG or AlexNet-era
  feature extraction conventions, not a modern vision-language model.
- **Dataset/category tested on:** Four Amazon categories: Baby, Sports and
  Outdoors, **Clothing, Shoes, and Jewelry** (the same category this
  project has used throughout, phases 1-11), and Electronics.
- **Metrics used to evaluate the system:** Confirmed directly from the full
  paper (fetched from `arxiv.org/html/2407.12338`): **Recall@K and NDCG@K
  at K=10 and K=20**, with an 8:1:1 train/validation/test split. Compared
  against MF-BPR, LightGCN, VBPR, MMGCN, SLMRec, LATTICE, BM3, MGCN, and
  MENTOR (the strongest baseline). One correction to this review's earlier
  assumption: **GALORE is not one of the head-to-head baselines in GUME's
  results table**, it is cited once, in the ablation discussion, as
  independent confirmation of a pattern GUME also observes (see limitation
  note below), not compared against directly.
- **Key number/result:** GUME beats MENTOR (its strongest baseline) on 3 of
  4 datasets, including **Clothing, Shoes, and Jewelry, the category this
  project uses throughout**: R@10 0.0703 vs. 0.0668, R@20 0.1024 vs. 0.0989
  (+3.54% relative), N@10 0.0384 vs. 0.0360, N@20 0.0466 vs. 0.0441 (+5.67%
  relative). Similar-sized gains on Sports (R@20 +2.28%) and Electronics
  (R@20 +3.82%). On Baby, the one honest exception, GUME actually ties or
  very slightly loses to MENTOR at R@20 (0.1042 vs. 0.1048). An ablation
  (Section 4.3) removes three components (Graph Enhancement/GE, Alignment/
  AL, User Modality/UM): removing the Alignment module causes the largest
  drop, then User Modality enhancement, then Graph Enhancement (smaller but
  consistent, concentrated in tail-item performance).
- **Stated weakness/limitation:** The paper has no dedicated limitations
  section, but the ablation discussion contains the closest thing to a
  self-reported trade-off, quoted directly: "Although removing graph
  enhancement can improve the recommendation performance for head items,
  the overall performance decreases due to the decline in tail item
  performance, which is consistent with the findings of GALORE [Luo et al.
  2023]." That is, GUME's own graph-enhancement component trades some
  head-item accuracy for tail-item gains, an explicit head/tail tension
  the paper acknowledges but doesn't resolve. Beyond that self-reported
  point, this review's own earlier framing still stands: GUME requires a
  full user-item interaction graph (structurally different from this
  project's item-only approach) and offers no inference-time
  controllability, its behavior is fixed once trained. On feature
  dimensions: confirmed directly, visual features are 4096-d and text
  features are 384-d, extracted "following the same setting mentioned in
  [Zhou 2023]" ("MMRec: Simplifying Multimodal Recommendation," ACM
  Multimedia Asia Workshops 2023) — GUME reuses the MMRec framework's
  standard preprocessing rather than extracting features itself, and the
  paper doesn't restate which underlying pretrained encoder MMRec's own
  4096-d visual features come from (that would require checking Zhou 2023
  directly, not done here, out of this task's scope).

---

## 5. Complementary Product Recommendation for Long-tail Products

Papso, 2023 | RecSys 2023

Adjacent problem, different mechanism, included for completeness on the
complementary-recommendation-for-sparse-data angle.

- **Technique/model used:** Transfer learning: complementary relations are
  learned on large, data-rich catalogs and transferred to small, sparse
  ones.
- **How it's used:** Addresses complementary product recommendation
  specifically for small and medium e-commerce platforms with limited
  interaction data, by reusing complementary-relation knowledge learned
  elsewhere rather than learning it from scratch on a sparse catalog. No
  visual embeddings and no controllability are involved, this solves an
  adjacent but distinct problem from this project's direction.
- **Dataset/category tested on:** **Not confirmed** in the source material
  available for this review.
- **Metrics used to evaluate the system:** **Not confirmed** in the source
  material available for this review.
- **Key number/result:** **Not confirmed** in the source material
  available for this review.
- **Stated weakness/limitation:** **Not confirmed** in the source material
  available for this review. Separately, and importantly: whether usable
  code exists for this method has **not yet been confirmed**, and per this
  review's own scope, this method should not be attempted as a
  reproduction target until that's actually checked.

---

## 6. Is It Really Complementary? Revisiting Behavior-based Labels for Complementary Recommendation

Sugahara, Yamasaki & Okamoto, 2024 | RecSys 2024

Not about long-tail exposure directly, included because it independently
validates a finding this project already reached on its own.

- **Technique/model used:** A quantitative re-evaluation methodology:
  behavior-based labels (BBLs, constructed from co-purchase logs, the same
  family of signal as Amazon's `also_buy`) are compared against manually
  annotated function-based labels (FBLs) for the same item pairs. A note on
  sourcing before the detail below: the ACM Digital Library full text
  (`dl.acm.org/doi/fullHtml/10.1145/3640457.3691705`) returned an HTTP 403
  and could not be fetched directly. What follows is drawn from two
  accessible secondary sources instead: the authors' own RecSys 2024
  conference presentation slides, and a closely related, same-author-group
  extension paper (Yamasaki, Sugahara, Nagi & Okamoto, "Function-based
  Labels for Complementary Recommendation: Definition, Annotation, and
  LLM-as-a-Judge," arXiv 2507.03945, submitted to Pattern Recognition
  Letters) that explicitly documents itself as "an extension of a previous
  study," citing the exact same IEICE technical report (Sugahara, Yamasaki,
  Nagi & Okamoto, 2024) that the RecSys slides cite as their own FBL
  annotation methodology's source. The dataset, annotation process, and
  category distribution below are therefore reasonably attributed to the
  same underlying FBL construction this RecSys 2024 paper draws on, not
  independently confirmed inside the RecSys paper's own five pages.
- **How it's used:** Manually annotated ground truth (function-based
  labels, judged directly from item functional relationships, independent
  of purchase history) is used as the reference against which
  behavior-based co-purchase labels are scored, measuring how often a
  co-purchase signal actually corresponds to a real functional
  complementary relationship.
- **Dataset/category tested on:** The **ASKUL dataset** (`askul.co.jp`), a
  Japanese B2B office-supplies and facilities e-commerce site, restricted
  to the "Office Supplies/Stationery" and "Household Goods/Kitchenware"
  categories specifically because random sampling across the full catalog
  produces mostly "unrelated" pairs. Item pairs were sampled from
  co-purchase logs using the D'Hondt method (an allocation method
  normally used for proportional election seats, repurposed here to
  balance sampling across item-category pairs by co-purchase count) to get
  category diversity, 2,000 pairs this way plus 800 pairs already flagged
  as complementary by site practitioners, for 2,800 pairs total submitted
  for annotation. Confirmed as the same underlying dataset the RecSys 2024
  paper's own presentation cites.
- **Metrics used to evaluate the system:** Confirmed from the RecSys
  presentation slides: **Hit@1**, comparing BBL-identified complementary
  pairs against the FBL ground truth. The extension paper reports a
  different, related evaluation (Macro-F1 for ML/LLM models trained to
  predict FBL labels themselves, not a BBL-vs-FBL comparison), so it isn't
  a second confirmation of the same Hit@1 number, it's evidence the same
  FBL dataset supports more than one kind of evaluation.
- **Key number/result:** The headline finding, confirmed via the
  presentation slides: **"the Hit@1 of BBL was below 0.5, indicating that
  complementary pairs identified by BBLs often did not have complementary
  relationships from the perspective of FBLs."** From the extension paper,
  additional confirmed detail on the underlying annotation: **9 functional
  relationship categories** were defined (same function/substitute;
  replenishable-with, both directions; must-combine-to-be-usable; combined
  becomes more useful, both directions and jointly; no relationship; and a
  ninth catch-all for relationships too complex to verbalize), refined
  through iterative pre-tests. **18 annotators** (7 undergraduate and 9
  graduate computer science students, 2 faculty) each labeled 400-500
  pairs, majority vote of 3 annotators per pair required, yielding a final
  clean dataset of **2,759 pairs** (41 of the original 2,800 excluded as
  not fitting any FBL definition). Label distribution: 14.9% substitute,
  21.4% complementary (split across the finer replenishment/combination
  subtypes), 48.0% unrelated, 10.8% too-complex-to-verbalize, with only
  4.8% outright annotator disagreement. Inter-annotator agreement (Fleiss'
  kappa) was 0.63 overall (moderate), lower (~0.4) for two of the
  finer-grained complementary subtypes.
- **Stated weakness/limitation:** Not applicable to the RecSys 2024 paper
  itself in the usual self-reported sense (its own text wasn't directly
  accessible to check), but this finding is itself effectively a stated
  weakness of the behavior-based labeling convention used across the
  field. What could be verified beyond the headline number: the extension
  paper found LLMs (used as an alternative, cheaper annotator) tend to
  systematically misclassify item pairs that humans label as "same
  function/substitute" or "combining makes both more useful" as "no
  relationship" instead, e.g. envelopes of different sizes, or sponges
  paired with bleach, cases where the relationship is real but weak or
  optional rather than necessary. That specific finding is about LLM-vs-
  human labeling confusion, not BBL-vs-FBL confusion, so it's a related but
  distinct result from this paper's own **not confirmed** headline claim
  about which kinds of pairs BBL most often mislabels; that specific
  breakdown wasn't found in either accessible source and is flagged here
  rather than inferred. Directly relevant to this project regardless: this
  paper independently confirms, through a different methodology (manual
  functional annotation vs. this project's own empirical phases 6 through
  8), the same conclusion this project already reached, that `also_buy`-
  style co-purchase data is a noisy mix, not a clean complement signal.
  Cite this explicitly wherever that earlier finding is discussed, it
  substantially strengthens that result's credibility with independent,
  published corroboration.

---

## Synthesis: the research gap this project will target

Across the systematic review's 71 papers and every current 2026 paper
found in this scan, no method offers inference-time steerable behavior.
Every method reviewed here commits to one fixed output once trained: the
systematic review's own six-family taxonomy (cluster, graph, deep
learning, multi-objective, traditional, other) contains no controllable
category at all. SAGERec and SAGE, the two current 2026 papers confirming
this is a live research area, are both user-representation or
user-sequence methods that require rich individual interaction histories,
a structurally different problem from an item-only approach. GUME is the
closest visual, long-tail-aware prior work found, it explicitly combines
multimodal item similarity with long-tail exposure on the same Amazon
Clothing, Shoes, and Jewelry category this project already uses, but it
requires a full user-item interaction graph, uses an older-generation
visual backbone (4096-d CNN features, not a modern vision-language model),
and has no inference-time controllability, its output is fixed once
trained.

This project's direction, a controllable mechanism, steerable between
accurate/popular and long-tail-favoring retrieval, operating purely on
items with no user interaction history required, built on a modern
vision-language backbone (SigLIP) already proven superior to older CNN
features in this project's own prior benchmarking (phases 1-4), is not
covered by any paper found in this review. That is the specific,
defensible contribution this project will target: not a vague novelty
claim, but a direct gap relative to GUME (item-only and backbone-modern
where GUME is graph-dependent and backbone-dated), relative to
SAGERec/SAGE (item-only where both require user interaction history), and
relative to the systematic review's own taxonomy (controllable where none
of the six reviewed approach families are).

## Do not do yet

Designing or implementing the actual controllable long-tail mechanism is
the next step, not this task, this document exists first as a stable,
written reference. Papso's transfer-learning method should not be
attempted as a reproduction target until its code availability is actually
confirmed, that check has not been done yet.

## Open follow-ups flagged by this review

**Update (follow-up pass):** GUME's metrics, headline results, and feature
dimensions are now confirmed directly from the full paper
(`arxiv.org/html/2407.12338`), and Sugahara et al.'s dataset and annotation
methodology are now confirmed via two accessible secondary sources (the
authors' own RecSys 2024 slides and a closely related same-author-group
extension paper) since the ACM full text itself returned a 403. See both
cards above for the sourcing caveats that come with that.

Fields still marked **not confirmed** are limited to the two brand-new 2026
papers (SAGERec, SAGE) and Papso's dataset/metrics/results/code
availability, none of which were in scope for the follow-up pass that
filled in GUME and Sugahara et al. Before citing any of those remaining
numbers in a report or building against them as a baseline, the actual
papers should be pulled and checked directly rather than treating this
review's placeholders as final.

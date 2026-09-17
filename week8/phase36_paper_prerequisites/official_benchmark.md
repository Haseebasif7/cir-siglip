# Phase 36: Official Polyvore Outfits Test Split -- Compatibility AUC and FITB

Released files: `nondisjoint/compatibility_test.txt` (20000 lines, 10000 positive) and `nondisjoint/fill_in_blank_test.json` (10000 questions; `answers[0]` is the correct candidate). 0 lines/questions skipped. These reference the same 10,000 `test.json` outfits the CIR benchmark is built from.

## Guards (secondary protocol must reproduce phase 9's recorded numbers before anything new is trusted)

| Guard | AUC | FITB | expected (AUC, FITB) | ok |
|---|---|---|---|---|
| raw_siglip_image_only | 0.7172 | 0.4843 | (0.7172, 0.4843) | yes |
| phase9_model_a | 0.9469 | 0.7031 | (0.9469, 0.7031) | yes |

## Primary protocol: CIR-native leave-one-out (each model scored through its own retrieval function)

AUC score of an outfit = mean over its items of s(outfit minus item, item); FITB = argmax over the 4 candidates of s(question, candidate). 107012 leave-one-out tasks. For CSA-Net the target category is the candidate's own category (9987/10000 FITB questions have all four answers in one category). Ensembles average member scores, never embeddings. Ties = FITB questions whose top score is shared (resolved to the lower index, as in phase 9).

| System | AUC | FITB acc. | FITB ties |
|---|---|---|---|
| Raw SigLIP, image only (untrained) | 0.6870 | 48.43% | 9 |
| Raw SigLIP, image + text (untrained, 1536-d) | 0.6420 | 43.24% | 8 |
| Phase 9 Model A (image-only projection, original) | 0.9352 | 70.31% | 11 |
| `ours_solo` -- Ours, mean-pooled projection, single model (phase 27 text_only) | 0.9567 | 72.95% | 8 |
| `ours_ens` -- Ours, mean-pooled projection, 10-seed ensemble (phase 28) | 0.9655 | 75.48% | 9 |
| `ot_solo` -- OutfitTransformer mechanism, frozen backbone, matched optimization, single model (phase 31/32 seed 42) | 0.9313 | 73.16% | 7 |
| `ot_ens` -- OutfitTransformer mechanism, frozen backbone, matched optimization, 3-seed ensemble (phase 32) | 0.9349 | 74.64% | 7 |
| `csa_ens` -- CSA-Net mechanism, frozen backbone, matched optimization + random negatives, 3-seed ensemble (phase 34) | 0.9349 | 70.51% | 7 |

**Published, same released nondisjoint split, different (fine-tuned) backbone -- not matched-condition:**

| Published system | AUC | FITB acc. | Source |
|---|---|---|---|
| Vasileva et al. 2018, type-aware embeddings (ResNet-18 fine-tuned) | 0.88 | 57.60% | ECCV'18; nondisjoint; already cited in week3/phase9 results_table.md |
| CSA-Net, Lin et al. 2020 (ResNet-18 fine-tuned end-to-end, no text) | 0.91 | 63.73% | arXiv 1912.08967, Table 1, Polyvore Outfits (nondisjoint) |
| OutfitTransformer, Sarkar et al. 2023 (ResNet-18 fine-tuned + SentenceBERT fc, text) | 0.93 | 67.10% | arXiv 2204.04812, Tables 1-2, nondisjoint; AUC 0.92 without text |

## Secondary protocol: phase 9's exact protocol (mean pairwise cosine of unconditioned item embeddings)

Not defined for CSA-Net (every CSA-Net item embedding is conditioned on a category pair). Reported to link to the numbers this project has cited since phase 9 and to show sensitivity to the aggregation choice.

| System | AUC | FITB acc. | FITB ties |
|---|---|---|---|
| Raw SigLIP, image only (guard) | 0.7172 | 48.43% | 9 |
| Raw SigLIP, image + text (untrained, 1536-d) | 0.6547 | 43.24% | 8 |
| Phase 9 Model A (guard) | 0.9469 | 70.31% | 11 |
| `ours_solo` | 0.9558 | 72.95% | 8 |
| `ours_ens` | 0.9647 | 75.52% | 9 |
| `ot_solo` | 0.9327 | 71.65% | 7 |
| `ot_ens` | 0.9406 | 72.24% | 9 |
| `csa_ens` | n/a | n/a | -- |

Interpretation is written in `phase36_notes.md` after reading these tables.

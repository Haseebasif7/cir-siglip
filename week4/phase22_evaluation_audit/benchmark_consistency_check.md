# Phase 22, Step 2: Benchmark Construction Consistency Across Evaluated Models

Checked directly against each evaluation script's own path-construction code (not assumed from the fact that all three are described as using "the CIR benchmark") -- resolves each script's `BENCHMARK_JSON` constant to an absolute path and hashes the file it actually points at.

| Model | Evaluation script | Resolved benchmark path | SHA-256 |
|---|---|---|---|
| Phase 9 (via phase 20's verification) | `week4/phase20_full_consolidation/scripts/01_verify_phase9.py` | `week4/phase12_controllable_modes/data/cir_benchmark.json` | `b1f4149e461c045d22bd3f2f402f13ff051dfdefd2a2b980767e8715944bd5d8` |
| CSA-Net (phase 13b, frozen SigLIP backbone) | `week4/phase13b_csa_net_siglip_backbone/scripts/03_csa_cir_eval.py` | `week4/phase12_controllable_modes/data/cir_benchmark.json` | `b1f4149e461c045d22bd3f2f402f13ff051dfdefd2a2b980767e8715944bd5d8` |
| OutfitTransformer (phase 14b run 2, random negatives) | `week4/phase14b_outfittransformer_category_negatives/scripts/04b_cir_eval_random_negatives.py` | `week4/phase12_controllable_modes/data/cir_benchmark.json` | `b1f4149e461c045d22bd3f2f402f13ff051dfdefd2a2b980767e8715944bd5d8` |

**Confirmed: all three evaluation scripts resolve to the exact same file on disk** (identical SHA-256 hash, 1 distinct path string(s) all pointing at the same underlying file via `Path.resolve()`). There is no possibility of a quietly diverged copy -- CSA-Net and OutfitTransformer are not scored against a separately reconstructed benchmark, they read the identical bytes phase 9's verification reads.

## Rank-counting logic in each evaluation script (direct excerpt, not paraphrased)

**Phase 9 / phase 20 (`cir_eval.py`, `evaluate_recall`)**:
```
        ranks = (sims >= target_sims[:, None]).sum(axis=1)
            hits[k] += int((ranks <= k).sum())
```

**OutfitTransformer (`04b_cir_eval_random_negatives.py`, `evaluate_recall`)**:
```
        ranks = (sims >= target_sims[:, None]).sum(axis=1)
            hits[k] += int((ranks <= k).sum())
```

**CSA-Net (`03_csa_cir_eval.py`, `evaluate_csa_recall`)**:
```
                rank = int((dist_avg <= target_dist).sum())
                    if rank <= k:
```

All three use the same convention: rank = count of candidates with a similarity (or, for CSA-Net, a smaller-is-better distance, inverted via `<=`) at least as good as the target's own; hit@K = rank <= K. No script silently uses a stricter or looser tie-breaking rule than the others.


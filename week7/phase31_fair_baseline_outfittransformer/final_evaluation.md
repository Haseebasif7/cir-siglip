# Phase 31, Step 6: Final Evaluation -- Test Benchmark, Once

**Scope note (budget-constrained)**: steps 4 (scale sweep) and 5 (10-seed ensembling) were not completed -- this project's Modal account hit a hard budget wall mid-phase (~$22.28 spent on phase 31 against ~$7 remaining). This is a SINGLE-MODEL result (step 3's fully-tuned winner), not an ensemble. See `phase31_notes.md` for the full disclosure.

Best configuration identified entirely through validation-benchmark comparisons (steps 1-3): `input_mode=image_text, lr=1.5e-4, batch_size=384, uniformity_weight=0.1, margin=0.2`, val Recall@10=0.1924. Evaluated here, exactly once, on the actual test CIR benchmark.

## Full progression

| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |
|---|---|---|---|---|
| Phase 14b original (single config) | 0.0588 | 0.1286 | 0.1809 | val_loss selection, no text |
| OutfitTransformer published | 0.0958 | 0.1796 | 0.2198 | different backbone, benchmark not verified |
| **Phase 31 strengthened, single model** | **0.1799** | **0.3111** | **0.3844** | text + selection fix + tuning, no ensemble |
| Phase 28 text ensemble (project's own best) | 0.1904 | 0.3267 | 0.4079 | 10-model ensemble |

## Derived comparisons

- vs. phase 14b original: +205.9% / +141.9% / +112.5% relative (beats it at every K).
- vs. OutfitTransformer published: 187.8% / 173.2% / 174.9% of published numbers.
- vs. phase 28's text ensemble (project's own best, 10-model): 94.5% / 95.2% / 94.2%.

n_total=29681 n_skipped=0.


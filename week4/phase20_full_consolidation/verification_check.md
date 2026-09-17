# Phase 20, Step 1: Phase 9 Model A Re-Verification

Re-ran phase 9's existing checkpoint (`week3/phase9_polyvore_compatibility/models/model_a_random_negs.pt`, a plain ProjectionHead: frozen SigLIP 768-d -> 256 -> 128, trained on real Polyvore outfit co-occurrence with random negatives, no dial, no attention, no additional mechanism -- this is exactly what "projection" means per the brief) directly against the current CIR benchmark (`week4/phase12_controllable_modes/data/cir_benchmark.json`), the same file used throughout this entire redirected sequence. No retraining -- this checkpoint is unchanged since phase 9.

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Cited (phase 12 onward) | 0.1317 | 0.2464 | 0.3216 |
| Re-measured here | 0.1317 | 0.2464 | 0.3216 |

n_total=29681, n_skipped=0.

**CONFIRMED: exact match.**

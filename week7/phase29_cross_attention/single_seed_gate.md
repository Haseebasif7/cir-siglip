# Phase 29: Single-Seed Gate (Step 3)

| Configuration | Val Recall@10 |
|---|---|
| Phase 28 text-only, single seed (42), mean-pool | 0.1819 |
| Phase 29 cross-attention, single seed (42) | 0.1228 |

Relative change: -32.48% (gate requires >= 2% to proceed to full ensemble; threshold = 0.1855).

**Decision: NO-GO -- stop before the ten-seed budget, per the brief explicit discipline. Cross-attention does not show a real single-seed signal over mean-pooling; the phase stops here and reports this as the answer.**

Learning rate used: 0.0005 (chosen in step 1's LR check over phase 28's original 0.001 -- see training_log.md).
Epochs to convergence: 3 (best at epoch 2, early stopping patience=4).
Wall time: 385s.

# Phase 18b, Step 5: Smoothness Extended to Two Dimensions (Re-Run)

Same method as phase 18's `06_smoothness_2d.py`.

## Axis 1 (substitute <-> complement), alpha2 held fixed

- Adjacent (|delta alpha1|=0.25): mean overlap = 0.7581
- Distant (|delta alpha1|=1.0): mean overlap = 0.4003
- **Gap = 0.3578** (phase 18: 0.3476)

## Axis 2 (relevance <-> tail-exposure), alpha1 held fixed

- Adjacent (|delta alpha2|=0.25): mean overlap = 0.7730
- Distant (|delta alpha2|=1.0): mean overlap = 0.4648
- **Gap = 0.3082** (phase 18: 0.2825)

## Diagonal (both alpha1 and alpha2 move together)

- Adjacent: mean overlap = 0.6909
- Distant: mean overlap = 0.3797
- **Gap = 0.3112** (phase 18: 0.2895)

## Comparison

| Mechanism | Gap |
|---|---|
| Phase 18, axis 1 | 0.3476 |
| Phase 18, axis 2 | 0.2825 |
| Phase 18, diagonal | 0.2895 |
| **Phase 18b, axis 1** | **0.3578** |
| **Phase 18b, axis 2** | **0.3082** |
| **Phase 18b, diagonal** | **0.3112** |

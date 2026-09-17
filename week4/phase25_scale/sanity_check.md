# Phase 25, Step 5: Sanity Check -- No Degenerate Collapse at Any Tested Scale

`mean_pairwise_cosine` (cosine similarity between random pairs of a 512-item sample of the model's own output embeddings, recomputed every epoch) is this project's standing collapse diagnostic -- a value approaching 1.0 would mean all embeddings are converging to nearly the same point (degenerate collapse), while a healthy, well-spread embedding space keeps this value low. This architecture has no competing objectives or adversarial dynamics (unlike the controllable-dial mechanisms elsewhere in this project), so collapse was not expected, but confirmed directly rather than assumed, per the brief.

## Observed values across every scale tested

| Configuration | Epoch 0 collapse | Final-epoch collapse | Range across training |
|---|---|---|---|
| Width=1024, lr=0.0005 (step 1 winner) | 0.161 | 0.010 (epoch 5) | 0.010 - 0.161 |
| Depth=2, lr=0.0005 | 0.268 | 0.007 (epoch 5) | 0.007 - 0.268 |
| Depth=3, lr=0.001 (largest model, worst recall) | **0.497** (highest observed anywhere) | 0.007 (epoch 8) | 0.004 - 0.497 |
| out_dim=512, lr=0.0005 (step 3 winner) | 0.137 | 0.009 (epoch 5) | 0.009 - 0.137 |

## Interpretation

**No collapse at any tested scale.** The highest single value observed across every configuration in this phase is 0.497 (depth=3's very first epoch, before training has meaningfully progressed) -- well below any threshold that would indicate degenerate collapse, and this value *decreases* over training rather than increasing, in every single run checked. The consistent pattern across all configurations is a moderate-to-high value at epoch 0 (reflecting the random initialization's own embedding spread before any training signal has shaped it) that drops quickly to a small, stable value (0.004-0.017) within the first few epochs, as the contrastive loss pushes embeddings apart. This is the expected, healthy trajectory for this loss family, and it holds identically whether the model is small (256-d hidden layer, the phase 23/24 baseline) or an order of magnitude larger (depth=3, ~3M parameters).

Notably, depth=3's higher epoch-0 collapse value (0.497 vs. 0.16-0.27 for the other configurations) is consistent with `overfitting_check.md`'s observation that deeper networks start from a less-differentiated embedding space and take a few extra epochs to de-collapse -- but this resolves during training in every case, never persists, and never approaches actual collapse (a value near 1.0). Scale is not a collapse risk for this mechanism; the risk scale introduces here is overfitting (`overfitting_check.md`), not embedding degeneracy.

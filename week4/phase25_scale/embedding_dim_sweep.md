# Phase 25, Step 3: Final Embedding Dimension Sweep

Using the winning width and depth from steps 1-2 (hidden_dims=[1024], i.e. depth=1, since depth only ever hurt), testing out_dim in {256, 512} instead of 128, with the same small learning-rate recheck. batch_size=256, weight_decay=0.0, tau=0.15, R=8 held fixed. Selected by validation-benchmark Recall@10, max_epochs=12/patience=4.

| out_dim | lr | val Recall@10 | best_epoch / epochs run | Parameters | Wall time |
|---|---|---|---|---|---|
| 128 (step 1/2 winner, for reference) | 0.0005 | **0.1656** | 1 / 6 | 918,656 | 668s |
| 256 | 0.0005 | 0.1627 | 1 / 6 | 1,049,856 | 695s |
| 256 | 0.001 | 0.1607 | 1 / 6 | 1,049,856 | 704s |
| 256 | 0.002 | 0.1611 | 1 / 6 | 1,049,856 | 701s |
| 512 | 0.0005 | 0.1646 | 1 / 6 | 1,312,256 | 759s |
| 512 | 0.001 | 0.1637 | 1 / 6 | 1,312,256 | 759s |
| 512 | 0.002 | 0.1601 | 1 / 6 | 1,312,256 | 740s |

**Neither out_dim=256 nor out_dim=512 beats the out_dim=128 winner (0.1656).** out_dim=512/lr=0.0005 comes closest (0.1646) but still falls short.

## Interpretation

A larger final retrieval embedding does not help here -- another negative-to-neutral finding, in the same direction as depth (though far less severe; the out_dim=512 results are close to, not dramatically below, the 128-d winner, unlike depth's sharp degradation). This is a genuine variable worth having tested separately (a bigger *space* similarity is computed in is not the same knob as a wider hidden layer), and the answer is that 128 dimensions were already enough for this mechanism at this data scale -- widening the retrieval space adds parameters without adding retrieval quality. The final architecture keeps out_dim=128, unchanged from phase 9/23/24.

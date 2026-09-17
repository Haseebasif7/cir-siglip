# Phase 25, Step 2: Depth Sweep

Using the winning width from step 1 (1024), testing one and two additional hidden layers (2 and 3 hidden layers total, each at width 1024), with a small learning-rate recheck at each depth. out_dim=128, batch_size=256, weight_decay=0.0, tau=0.15, R=8 all held fixed. Selected by validation-benchmark Recall@10, max_epochs=12/patience=4.

| Depth (hidden layers) | lr | val Recall@10 | best_epoch / epochs run | Parameters | Wall time |
|---|---|---|---|---|---|
| 1 (step 1 winner, for reference) | 0.0005 | **0.1656** | 1 / 6 | 918,656 | 668s |
| 2 | 0.0005 | 0.1515 | 1 / 6 | 1,968,256 | 733s |
| 2 | 0.001 | 0.1468 | 1 / 6 | 1,968,256 | 736s |
| 2 | 0.002 | 0.1379 | 1 / 6 | 1,968,256 | 759s |
| 3 | 0.0005 | 0.1274 | 1 / 6 | 3,017,856 | 799s |
| 3 | 0.001 | 0.1183 | 4 / 9 | 3,017,856 | 1252s |
| 3 | 0.002 | 0.1139 | 5 / 10 | 3,017,856 | 1301s |

**Depth hurts, substantially and monotonically.**

## Interpretation

This is a clean, unambiguous negative finding, not a close call:

- Every depth-2 and depth-3 configuration scores below the depth-1 winner (0.1656), and below every single width-1024/depth-1 learning-rate variant tested in step 1 (all of which scored 0.160-0.166).
- The degradation is monotonic with depth: depth=2's best (0.1515) is worse than depth=1's worst-lr result (0.1603 at lr=0.001), and depth=3's best (0.1274) is worse still than depth=2's worst (0.1379).
- This is not a learning-rate artifact -- all three learning rates were tried at each depth, and none rescued depth=2 or depth=3 to anywhere near depth=1's performance.
- See `overfitting_check.md` for the mechanistic diagnosis of *why* depth hurts (a mix of overfitting and a slower initial de-collapse, not simply "needs more epochs").

No further depth was tested beyond 3 hidden layers, per the brief's own instruction not to keep scaling up in the direction of a clear negative finding.

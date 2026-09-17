# Phase 27: Training Log

All three runs used week 6's fixed hyperparameters (`lr=0.001, batch_size=256, weight_decay=0.0, tau=0.15, r_neg=8, hidden_dims=[1024], out_dim=128`), single seed (42, Modal's default), early stopping on validation-benchmark Recall@10 with `patience=4, min_delta=0.0005`. All three ran the full patience budget (5 epochs, best at epoch 0) before stopping.

## text_only (`use_category="none"`, in_dim=1536)

| Epoch | Train loss | Val Recall@10 | Val Recall@30 | Val Recall@50 |
|---|---|---|---|---|
| 0 | 4.7429 | **0.1819** | 0.3169 | 0.3950 |
| 1 | 4.3440 | 0.1760 | 0.3097 | 0.3896 |
| 2 | 4.0579 | 0.1623 | 0.2924 | 0.3719 |
| 3 | 3.8622 | 0.1561 | 0.2815 | 0.3584 |
| 4 | 3.7206 | 0.1492 | 0.2729 | 0.3503 |

Best: epoch 0, val Recall@10=0.1819. Wall time: 626s. Recall declines monotonically after epoch 0 even as training loss keeps falling -- the model is already at its best generalizing point after one epoch and overfits from there.

## text_category_learned (`use_category="learned"`, in_dim=2304)

| Epoch | Train loss | Val Recall@10 | Val Recall@30 | Val Recall@50 |
|---|---|---|---|---|
| 0 | 2.9857 | **0.1173** | 0.2281 | 0.2993 |
| 1 | 2.8081 | 0.0936 | 0.1869 | 0.2473 |
| 2 | 2.6744 | 0.0701 | 0.1424 | 0.1944 |
| 3 | 2.5657 | 0.0536 | 0.1163 | 0.1630 |
| 4 | 2.4820 | 0.0454 | 0.0984 | 0.1375 |

Best: epoch 0, val Recall@10=0.1173. Wall time: 645s. Same monotonic-decline shape as `text_only`, but far steeper, and from a much lower starting point.

## text_category_siglip_phrase (`use_category="siglip_phrase"`, in_dim=2304)

| Epoch | Train loss | Val Recall@10 | Val Recall@30 | Val Recall@50 |
|---|---|---|---|---|
| 0 | 2.9785 | **0.1203** | 0.2328 | 0.3060 |
| 1 | 2.8349 | 0.1106 | 0.2121 | 0.2802 |
| 2 | 2.7258 | 0.0849 | 0.1745 | 0.2359 |
| 3 | 2.6301 | 0.0703 | 0.1478 | 0.2020 |
| 4 | 2.5530 | 0.0593 | 0.1215 | 0.1697 |

Best: epoch 0, val Recall@10=0.1203. Wall time: 647s. Nearly identical shape and magnitude to the learned-table variant -- the category-encoding *method* barely matters; both collapse the same way.

## A pattern worth flagging directly: training loss vs. validation Recall move in OPPOSITE directions for the category variants

Both category-conditioned variants show markedly *lower* absolute training loss than `text_only` (~2.5-3.0 vs. ~3.7-4.7) while scoring *much worse* on held-out Recall@10, and the gap between training-loss improvement and Recall@10 collapse widens every epoch. That is the textbook signature of the model exploiting a shortcut available during training that doesn't exist at evaluation time -- see `phase27_notes.md` for the specific mechanism this points to (training negatives are not category-restricted, but evaluation candidate pools are).

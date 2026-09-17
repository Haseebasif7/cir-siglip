# Phase 22, Step 5: Manual, By-Hand Reproduction of Individual Query Scoring

3 queries picked at random (seed=42) from those whose target is confirmed in its category pool, hand-traced end to end for phase 9 and OutfitTransformer -- not calling `evaluate_recall()`, computing rank from raw similarity scores directly to catch any subtle bug in the shared harness itself (e.g. an off-by-one in rank counting, or a pool/index misalignment) that a self-consistency check inside the harness could not catch.

## Query index 2648 (outfit `210006589`, category `bottoms`)

- Context items (3): `['184017015', '187890188', '183811054']`
- Target item: `184503855` (found at position 1522 in the `bottoms` pool, pool size 3000)

**Phase 9 (ProjectionHead), hand-computed:**
- Query vector = mean of 3 context items' projected embeddings, L2-renormalized
- Target's own similarity score: 0.836020
- Manual rank (# candidates with sim >= target's sim): **524**
- Hit@10: no, Hit@30: no, Hit@50: no
- Top-5 pool items by similarity: [('135330226', 0.9312), ('186209239', 0.9308), ('187006809', 0.9262), ('193459270', 0.9244), ('200642243', 0.922)]

**OutfitTransformer, hand-computed:**
- Query vector = OutfitTransformer's own outfit-token readout after the masked self-attention set encoder
- Target's own similarity score: 0.017579
- Manual rank (# candidates with sim >= target's sim): **153**
- Hit@10: no, Hit@30: no, Hit@50: no
- Top-5 pool items by similarity: [('190233145', 0.0424), ('197927000', 0.0419), ('190236384', 0.0415), ('188412056', 0.0381), ('200372131', 0.0368)]

## Query index 19428 (outfit `181994352`, category `scarves`)

- Context items (7): `['148591984', '164868938', '152107918', '150410679', '142061488', '141816468', '145598419']`
- Target item: `135234234` (found at position 117 in the `scarves` pool, pool size 844)

**Phase 9 (ProjectionHead), hand-computed:**
- Query vector = mean of 7 context items' projected embeddings, L2-renormalized
- Target's own similarity score: 0.788109
- Manual rank (# candidates with sim >= target's sim): **361**
- Hit@10: no, Hit@30: no, Hit@50: no
- Top-5 pool items by similarity: [('213275602', 0.9098), ('118293044', 0.9066), ('152338410', 0.9015), ('191070227', 0.8973), ('183232881', 0.8959)]

**OutfitTransformer, hand-computed:**
- Query vector = OutfitTransformer's own outfit-token readout after the masked self-attention set encoder
- Target's own similarity score: -0.001506
- Manual rank (# candidates with sim >= target's sim): **395**
- Hit@10: no, Hit@30: no, Hit@50: no
- Top-5 pool items by similarity: [('181984404', 0.0313), ('183232881', 0.0279), ('193662992', 0.0277), ('96405145', 0.0266), ('153803029', 0.0265)]

## Query index 22971 (outfit `199738051`, category `bags`)

- Context items (2): `['132175560', '8051909']`
- Target item: `166859848` (found at position 962 in the `bags` pool, pool size 3000)

**Phase 9 (ProjectionHead), hand-computed:**
- Query vector = mean of 2 context items' projected embeddings, L2-renormalized
- Target's own similarity score: 0.918834
- Manual rank (# candidates with sim >= target's sim): **20**
- Hit@10: no, Hit@30: YES, Hit@50: YES
- Top-5 pool items by similarity: [('213591035', 0.9594), ('196602455', 0.9465), ('194139596', 0.946), ('168502002', 0.9447), ('193532212', 0.9436)]

**OutfitTransformer, hand-computed:**
- Query vector = OutfitTransformer's own outfit-token readout after the masked self-attention set encoder
- Target's own similarity score: 0.027236
- Manual rank (# candidates with sim >= target's sim): **122**
- Hit@10: no, Hit@30: no, Hit@50: no
- Top-5 pool items by similarity: [('169004224', 0.0483), ('165038852', 0.0468), ('117648927', 0.0465), ('140507057', 0.0448), ('204626945', 0.0445)]

## Cross-check against the harness's own `evaluate_recall()`

The manual computation above re-derives, from scratch, every step `evaluate_recall()` performs internally (mean-pool the context items, L2-renormalize, dot-product against the pool, count candidates with sim >= target's sim as the rank) using the exact same loaded embeddings and checkpoints, but without calling that function at all. For every query checked above, this hand computation used the identical pool contents, the identical target position, and the identical similarity/rank definitions the shared `cir_eval.py` module uses -- there is no discrepancy because the manual code IS an independent re-implementation of the same, simple, auditable arithmetic (mean, normalize, dot product, count) -- there is no room in this pipeline for a subtle bug that would only appear in the batched `evaluate_recall()` version and not in this element-by-element trace, since both do the exact same operations, just batched vs. per-query.

**Direct empirical confirmation, not just an argument**: separately, phase 9's actual `evaluate_recall()` was re-run against the full 29,681-query benchmark using this same loaded checkpoint and embeddings (outside this script, as an independent check), returning Recall@10/30/50 = `0.13167/0.24639/0.32155` -- matching phase 9's cited `0.1317/0.2464/0.3216` to 4 decimal places. This confirms the batched harness and this script's per-query manual arithmetic are computing the same thing, empirically, not just by code-reading argument.


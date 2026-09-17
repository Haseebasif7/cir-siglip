# Phase 9, Step 2: Positive Edge Summary

- Train: 1373702 directed positive edges from 53306 outfits (0 item-slots skipped, missing image)
- Val: 129850 directed positive edges from 5000 outfits (0 item-slots skipped, missing image)

## Cross-category check (train edges), verified against actual semantic_category labels

The brief's assumption: since outfits are typically assembled one item per role (a top, a bottom, shoes, etc.), most also_buy-style pairs here should already be cross-category, unlike Amazon's data -- checked directly rather than assumed:

| Bucket | Count | % |
|---|---|---|
| Cross-category (different semantic_category) | 1313476 | 95.6% |
| Same-category (same semantic_category) | 60226 | 4.4% |
| Unknown type on one/both ends | 0 | 0.0% |

**Confirmed: the assumption holds strongly** (95.6% cross-category) -- unlike Amazon's also_buy edges (74.1% same-category, phase 8), Polyvore's co-outfit pairs are overwhelmingly already cross-category by construction. No category-level correction (like phase 8's heterogeneous-dyad filtering) is needed here -- the raw co-outfit pairs ARE the heterogeneous-dyad-equivalent signal already.


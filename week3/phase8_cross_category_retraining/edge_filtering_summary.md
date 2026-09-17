# Phase 8, Step 2: Heterogeneous-Dyad Edge Filtering Summary

Starting point: phase 7's 76293 positive also_buy edges (both endpoints in the shared 24,719-product cleaned pool).

| Bucket | Count | % of phase 7 total |
|---|---|---|
| Heterogeneous (different type, kept as phase 8 positives) | 16715 | 21.9% |
| Same type (excluded -- near-duplicate/substitute-like) | 56509 | 74.1% |
| Unknown type on one/both ends (excluded, conservative) | 3069 | 4.0% |

**16715 heterogeneous-dyad edges clears the workable floor (~3000) comfortably -- no pool expansion was needed.**

Re-split 90/10: 15044 train edges, 1671 val edges.

## Reading

The majority of also_buy edges in this catalog (74.1%) connect products of the SAME fine-grained type -- consistent with phases 6 and 7's finding that also_buy is dominated by near-duplicate/substitute-like pairs (same item, different color/size/listing) rather than genuine cross-category complements. The remaining 21.9% heterogeneous slice is what this phase trains on -- smaller than phase 7's full edge set, but still a substantial 16715-edge training signal.


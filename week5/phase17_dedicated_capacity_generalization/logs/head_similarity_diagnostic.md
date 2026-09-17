# Phase 17: Head Similarity Diagnostic

Adapted equivalent of phases 12/12b/12c's per-item cosine(z_sub, z_comp) check on their shared-trunk architecture -- here there is no single shared projection output, so this compares the two dedicated heads' own OUTPUT embeddings for the same 5000 sample items directly (same method phase 16d used on the relevance/tail axis).

**Mean per-item cosine similarity between substitute_head's and complement_head's outputs, same items: -0.0492 (std 0.0555)**

## Comparison across the sequence

| | Phase 12 | Phase 12b | Phase 12c | Phase 16d (relevance/tail heads) | Phase 17 (substitute/complement heads) |
|---|---|---|---|---|---|
| Cosine similarity | 0.8288 | 0.5844 | 0.3000 | -0.0147 | **-0.0492** |

(Phases 12/12b/12c measure cosine between the shared-trunk-derived substitute and complement PROJECTIONS of the same items; phase 16d and this phase measure cosine between two structurally independent HEADS' outputs -- both are the same underlying question (how differentiated are the two modes' representations of the same item), adapted to whichever architecture is being diagnosed.)

## Secondary, structural comparison: the two heads' weight matrices

- cosine(substitute_head.weight, complement_head.weight), flattened: -0.0069

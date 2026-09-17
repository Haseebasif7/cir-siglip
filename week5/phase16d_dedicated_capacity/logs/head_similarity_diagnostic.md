# Phase 16d: Head Similarity Diagnostic

Adapted equivalent of phases 16/16c's mode-vector cosine similarity check -- there's no single mode-vector parameter here, so this compares the two dedicated heads' own OUTPUT embeddings for the same 2000 sample items directly.

**Mean per-item cosine similarity between relevance_head's and tail_head's outputs, same items: -0.0147 (std 0.0854)**

## Comparison across the sequence

| | Phase 16 (mode vectors) | Phase 16c (mode vectors) | Phase 16d (head outputs) |
|---|---|---|---|
| Cosine similarity | 0.5525 | 0.0590 | -0.0147 |

## Secondary, structural comparison: the two heads' weight matrices

- cosine(relevance_head.weight, tail_head.weight), flattened: -0.0026
- relevance_head weight norm: 24.5602, tail_head weight norm: 24.1802

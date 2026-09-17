# Phase 18, Step 1: Tail-Restricted Nearest-Neighbor Lookup -- Coverage Report

Precomputed raw SigLIP top-50 nearest neighbors, restricted to TAIL-TIER items only (phase 3's tier definition), for all 24719 items in phase 7's Amazon pool. This is the new training signal the substitute-tail-exposure corner needs -- items that are both visually similar to an anchor AND catalog-tail (low `also_buy`/`also_viewed` reference count).

- **Tail-tier items available as neighbor candidates: 11190/24719 (45.3%)** -- confirmed directly against phase 3's `popularity_lookup.csv`, not assumed. This matches phase 16c's own item-level tail composition finding for this same pool (45.3%).
- Build time: 1.3s (0.02 min), 18375 items/sec, device=mps. Chunk size 4096, 7 chunks -- no subsampling needed.
- Output: `data/tail_nn_lookup.npz` (10.4 MB): `indices` (int32, (24719,50), global indices into phase 7's embedding array), `sims` (float32, raw cosine similarity, the ranking-distillation teacher), `is_tail` (bool, ({n_items},), tier membership per pool item).

## Coverage checks

- **Anchors that needed padding (fewer than 50 distinct tail-tier neighbors found): 0/24719 (0.00%)**. With 11190 tail-tier candidates available and only K=50 needed per anchor, padding is expected to be at most a handful of edge cases (e.g. an anchor whose only near-duplicate embeddings happen to be non-tail), not a systemic problem -- confirmed here rather than assumed.
- Self-inclusion check (sampled ~2000 items): 0/2000 had themselves in their own top-50 tail-restricted neighbor list (expected 0).
- Non-tail leakage check (sampled ~2000 items): 0/2000 had at least one non-tail-tier item in their top-K list (expected 0 -- the restriction is a hard column mask, not a soft preference).
- Neighbor similarity range: min=0.3240, max=1.0000, mean=0.6838.

## Verdict

Coverage is adequate: every anchor in the pool has (effectively) a full set of 50 distinct tail-tier visual neighbors to train against, with negligible padding. Proceeding to step 2 (the unrestricted substitute-relevance lookup) and then training -- no supplementary tail sample or reduced K was needed.


# Phase 18, Step 2: Unrestricted Nearest-Neighbor Lookup (Substitute-Relevance) -- Coverage Report

Precomputed raw SigLIP top-50 nearest neighbors, unrestricted (phase 12c's original method, never previously run on Amazon data), for all 24719 items in phase 7's Amazon pool. This is the substitute-relevance corner's ranking-distillation teacher.

- Build time: 2.6s (0.04 min), 9365 items/sec, device=mps.
- Output: `data/unrestricted_nn_lookup.npz` (10.4 MB).
- Self-inclusion check (sampled ~2000 items): 0/2000 (expected 0).
- Neighbor similarity range: min=0.3497, max=1.0000, mean=0.7302.

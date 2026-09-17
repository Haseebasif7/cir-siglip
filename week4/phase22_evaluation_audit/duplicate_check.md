# Phase 22, Step 4: Duplicate / Near-Duplicate Images in the CIR Benchmark's Candidate Pools

Checked directly against the actual image files in `week3/phase9_polyvore_compatibility/data/images/` and the raw frozen SigLIP embeddings in `siglip_base.npz` -- not assumed carried over from the phases 6-8 Amazon-data dedup work, which was a different dataset (Amazon Clothing) entirely and never touched Polyvore images.

## Check 1: exact byte-identical images among pool items

15 groups of byte-identical images found among the 26494 unique pool items, involving 30 items total (0.11% of pool items).

Largest groups (up to 10 shown):

| Group size | Example item ids |
|---|---|
| 2 | 121183138, 150981589 |
| 2 | 121645395, 132599105 |
| 2 | 135696099, 164937050 |
| 2 | 140971349, 156534847 |
| 2 | 142455205, 147343253 |
| 2 | 156530097, 164937039 |
| 2 | 157851550, 185833243 |
| 2 | 158230687, 190549458 |
| 2 | 166537311, 213098179 |
| 2 | 184112987, 185421373 |

## Check 2: does any query's TARGET item byte-match one of its own CONTEXT items? (the critical leakage check)

**0 / 29681 queries (0.000%)** have a target item that is byte-for-byte identical to one of the items already given as query context.

## Check 3: near-exact duplicates within each category pool (raw SigLIP cosine > 0.995)

| Category | Pool size | Near-duplicate pairs |
|---|---|---|
| bags | 3000 | 1 |
| shoes | 3000 | 0 |
| jewellery | 3000 | 4 |
| all-body | 3000 | 5 |
| tops | 3000 | 2 |
| bottoms | 3000 | 9 |
| outerwear | 3000 | 8 |
| sunglasses | 2202 | 11 |
| accessories | 1254 | 1 |
| hats | 1194 | 3 |
| scarves | 844 | 0 |

**44 total near-duplicate pairs** across all pools (includes the exact byte-identical pairs from Check 1, since those trivially also pass a >0.995 cosine threshold).

## Check 4: near-exact (not just byte-exact) target/context leakage

**0 / 29681 queries (0.000%)** have a target item with raw-SigLIP cosine similarity > 0.995 to one of its own context items -- this superset includes Check 2's byte-identical cases plus any resized/recompressed near-duplicates a byte hash would miss.

## Verdict

**Duplicate/near-duplicate contamination is real but negligible at benchmark scale.** 0 queries (0.000%) are trivially solvable via target/context duplication, and near-duplicate pairs are a small fraction of total pool items. At this rate, duplicate leakage could not plausibly explain phase 9's roughly 2x advantage over the next-best 'mine' configuration (phase 13b's CSA-Net reproduction) -- removing 0.000% of queries would not move Recall@10 by an amount close to the gap being explained. This is reported as a known minor imperfection in the benchmark construction, not a fatal flaw.


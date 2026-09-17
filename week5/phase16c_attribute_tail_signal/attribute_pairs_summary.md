# Phase 16c: Attribute-Based Positive Pairs Summary

## How pairs were constructed

Attribute used: fine-grained category (phase 8's `product_types.json`, breadcrumb index 3, 711 distinct types across the pool, no new fetch needed). Brand was attempted (`01_extract_brand.py`) but stopped as impractically slow after the source throttled to ~270 lines/s (~2.5-3hr projected for the full pass) -- see `logs/brand_extraction_report.md`. Category alone proved sufficient, see the tier composition below.

Groups used (size >= 2): **214**, covering **22934** pool items.

Per-group cap: 150 pairs, stratified 45% tail-tail / 35% tail-other / 20% other-other (deliberately oversampling tail-tier involvement -- unmet quota in the tail strata is NOT silently backfilled with more other-other pairs beyond the redistribution rule below, so small or tail-sparse groups just produce fewer total pairs rather than diluting the tail-inclusive design).

## Pool item-level tier composition (context)

Phase 7's 24719-item pool, by item (not by also_buy edge target, which is the distribution phase 16b hit the wall on): head=9961 (40.3%), mid=3568 (14.4%), tail=11190 (45.3%).

## Resulting attribute-pair edge set

- Total directed edges: **37922** (34130 train / 3792 val), from 18961 undirected pairs across 214 groups.
- Unique items involved (source or target): **11273**

### Tier composition ACHIEVED (the number that actually matters for this phase's premise)

| | as ANCHOR (source) | as TARGET | involved (either side, unique items) |
|---|---|---|---|
| head | 12748 (33.6%) | 12748 (33.6%) | 3877 (34.4%) |
| mid | 5871 (15.5%) | 5871 (15.5%) | 1737 (15.4%) |
| tail | 19303 (50.9%) | 19303 (50.9%) | 5659 (50.2%) |

**Compare directly to phase 16b's wall**: only 0.16% of also_buy edge targets were tail-tier there (125/76,293). Here, **50.9% of edge targets are tail-tier** -- confirming attribute-based pairing genuinely sidesteps that structural problem rather than just working around it superficially.

## Step 2: overlap check against phase 16/16b's also_buy edges

- Also_buy directed edges (phase 16/16b's signal): 76293
- Attribute-based directed edges (this phase's signal): 37922
- Directed overlap (exact same (source, target) pair in both): **660** (1.740% of attribute pairs, 0.865% of also_buy pairs)
- Undirected overlap (same two items paired, either direction): **448** (2.363% of attribute pairs)

**Verdict: genuinely different signal.** Overlap is negligible (1.740% of attribute pairs also appear as also_buy edges) -- this is not a disguised repeat of phases 16/16b's behavioral signal, it's drawing positive pairs from a structurally different source (shared fine-grained category, not co-purchase history).

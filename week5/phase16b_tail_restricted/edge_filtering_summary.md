# Phase 16b, Step 1: Tail-Tier Edge Filtering

Source: `week3/phase7_learned_compatibility/data/positive_edges.json` -- 76293 total also_buy edges (phase 7's full, unrestricted set, same one phase 16's relevance mode trains on).

## Tier breakdown of edge TARGETS (the restriction variable)

| Tier | Edge count | % of total |
|---|---|---|
| head | 74603 | 97.78% |
| mid | 1565 | 2.05% |
| tail | 125 | 0.16% |
| unknown (not in phase 3's tier lookup) | 0 | 0.00% |

## Tail-tier-restricted edge set (this phase's tail-exposure training signal)

- Total tail-tier edges: **125**
- Train split: **117**
- Val split: **8**
- Unique anchor (source) items: **125**
- Unique tail-tier target items: **37**

Minimum bar for 'meaningful to train on' (this phase's own pre-declared threshold): 1000 train edges.

**VERDICT: INSUFFICIENT.** 117 train edges falls short of the 1000-edge bar -- per the brief's explicit instruction, stopping here and reporting this directly rather than training on a thin signal.

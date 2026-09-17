# Phase 16c: Brand Attribute Extraction -- Stopped, Impractically Slow

Attempted a single retry-wrapped streaming pass over the raw Amazon metadata
(`meta_Clothing_Shoes_and_Jewelry.json.gz`, the same source used throughout
this project) to extract `brand` for phase 7's 24,719-item training pool,
matching phase 3's own streaming convention. `brand` is confirmed present
and populated in this source (a live 2,000-record sample check found it
present and nonempty in 1,390/2,000 = 69.5% of records), so this isn't a
"field doesn't exist" problem.

**Stopped after ~88 seconds / 121,081 records streamed.** The download rate
collapsed from >10,000 lines/s in the first few seconds to ~270 lines/s by
the time it was stopped -- at that throttled rate, completing the full
~2.68M-record pass would take on the order of **2.5-3 hours**, impractical
for what would only ever be a secondary, supplementary attribute signal.
This is reported directly per the brief's own instruction (report rather
than proceed on an impractical/incomplete fetch without flagging it),
rather than waiting indefinitely or silently substituting a weaker
heuristic (e.g. guessing brand from the title string).

**This does not block the phase.** Fine-grained category (phase 8's
already-built `product_types.json`, breadcrumb index 3, no new fetch
needed) independently covers 98% of the pool (24,223/24,719 items in
groups of size >= 2) across 215 usable groups, and the pool's item-level
tier composition (45.3% tail / 40.3% head / 14.4% mid, computed directly
against phase 3's tier lookup) is already healthy without any deliberate
oversampling -- category alone is sufficient to build a genuinely
tail-inclusive attribute-based training signal, so brand is dropped as a
secondary signal for this phase rather than substituted with something
noisier. See `attribute_pairs_summary.md` for the actual pair construction.

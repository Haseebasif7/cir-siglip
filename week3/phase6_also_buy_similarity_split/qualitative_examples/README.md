# Phase 6, Step 3: Qualitative Check on the Extremes (SigLIP)

Grids: `highest_similarity_siglip.png` (10 highest-similarity also_buy pairs),
`lowest_similarity_siglip.png` (10 lowest-similarity also_buy pairs). Raw
titles/categories/similarities for both groups: `extremes_summary_siglip.json`.

## Highest-similarity pairs: dominated by literal duplicate photos, not "near-duplicate products"

All 10 highest-similarity pairs score cosine similarity = 1.0000 (to 4 decimal
places). Checked the actual image files behind them (MD5 hash): **all 10 are
byte-for-byte identical image files.** Examples: two different Kiwi Shoe
Polish ASINs (different pack sizes) sharing the exact same product photo; two
different Eagle Creek packing-cube ASINs sharing the exact same photo; two
French Toast school-uniform ASINs (a boys' shirt and a girls' blouse -- same
brand/line, different garment) sharing the exact same photo.

This is a real but narrow pattern: of 9,450 edges, 61 (0.65%) score
similarity > 0.999, and **all 61 turn out to be byte-identical images** --
almost certainly the same seller/catalog photo reused across size, color, or
pack-size variants of what is often (but not always -- see the boys'/girls'
shirt example) the same underlying product. Between 0.95 and 0.999 there are
48 more edges that are visually near-identical but *not* byte-identical (i.e.
genuine close-variant photos, not exact file reuse) -- still a small slice
(1.15% of all edges, 0.95+) of the overall distribution.

**Reading**: the very top of the similarity distribution is not cleanly
"genuine near-duplicate/close-variant products" in the sense the phase set
out to check -- it's substantially a byte-level image-reuse artifact of how
this catalog's product photography is organized (variant listings sharing one
photo). This doesn't invalidate the broader distribution (99%+ of edges are
not in this extreme-duplicate regime) but it does mean a similarity-based
substitute label should not be validated using the very top of the
distribution as its "proof" case -- that regime is measuring photo reuse, not
visual similarity between genuinely distinct product photos.

## Lowest-similarity pairs: mostly a data-quality artifact, one genuine complement example

6 of the 10 lowest-similarity pairs involve at least one product whose title
is a corrupted scrape artifact -- literally the JavaScript snippet
`var aPageStart = (new Date()).getTime();` instead of a real product title.
Checked further: this affects 16 of 1,872 products in the sample (0.85%),
concentrated in Costumes & Accessories. Visually, the image behind this
corrupted-title entry is not a product photo at all -- it's a **sizing-chart
graphic** (a table of measurements), not a picture of a costume. These pairs'
low similarity is an artifact of comparing a real product photo against a
non-photo image, not a meaningful "these are visually different but
functionally paired" case.

Of the remaining (non-corrupted) low-similarity pairs:
- Two pairs of different, visually distinct Halloween/pirate/ninja costume
  products (e.g. "Rubie's Brotherhood of the Dragon" vs "California Costumes
  Stealth Ninja") -- these read more like **alternative costume choices**
  (substitute-like: same purchase occasion, different specific costume) than
  functional complements.
- One pair, **Smiffy's Women's Pirate Wig vs Underwraps Women's Pirate
  Ruffled Blouse**, is a genuine, clean example of a functional complement:
  visually unrelated items (a wig, a blouse) that plausibly get bought
  together to complete a pirate costume.
- One pair (Tarrago Self Shine Cream Kit vs KIWI Deluxe Shine Kit) is two
  different-brand shoe-shine products -- same functional category, different
  packaging/appearance; reads as a near-substitute rather than a complement.

**Reading**: the low-similarity tail does contain at least one clean
complement example, but the qualitative signal here is weaker and noisier
than hoped -- most of the extreme-low bucket is either a data-quality
artifact (corrupted title/non-photo image) or "different item, same general
purchase occasion" rather than a clean functional-complement pattern. This
should be reported honestly rather than treated as strong qualitative
confirmation of the complement hypothesis.

# Phase 11: Blended Similarity and Diversity-Forcing Retrieval — Notes

## Context

Two open questions motivated this phase. First, nothing in this project had ever
combined raw SigLIP similarity with a learned compatibility signal into one actual
retrieval output — phases 7-10 only ever evaluated similarity and compatibility
separately. Second, phase 9's Polyvore-trained model carries a documented
color-matching shortcut, but only a handful of manually inspected examples backed
that finding; it was an open question whether that shortcut is actually harmful in
practice or whether it incidentally produces useful diversity anyway. This phase
builds the missing combination mechanism (weighted blending, and a separate
rule-based diversity-forcing config) and settles both questions with full-sample
measurement (all 1,872 queries) rather than a handful of hand-picked cases.

No new training happened. Everything used already existed: raw SigLIP embeddings,
phase 8 Model A's checkpoint, and phase 9 Model A's checkpoint — this phase only
projects, blends, and evaluates.

## Step 1: gathering the pieces — clean, sanity-checked

Raw SigLIP embeddings were reused directly from `week2/phase1b_category_balanced/`.
Phase 8 Model A's and phase 9 Model A's projected embeddings were recomputed by a
frozen forward pass of each existing checkpoint over the same raw embeddings (no
training). A sanity check — recomputed raw SigLIP Hit Rate@5/10 and Precision@5/10
against the existing reported baseline (0.502/0.578/0.191/0.144) — passed before
anything downstream was trusted. Saved to `data/embeddings_bundle.npz` for reuse
across steps 2-5.

## Steps 2-4: blended and diversity-forcing configurations, evaluated three ways

Ten configurations were evaluated: raw SigLIP alone, phase 8 alone, phase 9 alone,
six blends (`blended = alpha * raw_similarity + (1-alpha) * compatibility_similarity`,
alpha in {0.3, 0.5, 0.7}, done separately for phase 8's and phase 9's compatibility
signal — not mixed together), and one diversity-forcing configuration (top-4 items
by raw SigLIP similarity, plus one slot reserved for the highest-raw-similarity item
of a different fine-grained type than the query, using phase 8's type index 3 and
its documented caveats).

**A construction detail worth stating explicitly**: the brief only specifies the
diversity-forcing rule for a top-5 list (top-4 + 1 forced slot). To report
Hit Rate@10 for this configuration too, this phase extends it as literally as
possible: the single forced slot stays at rank 5, and ranks 6-10 continue filling in
from the plain raw-SigLIP ranking (excluding anything already used) — one diversity
slot total, not a second one invented for the second half of the list. This choice
turns out to matter (see the diversity-metric discussion below).

**Diversity-forcing construction stats**: of 1,872 queries, 1,732 got a genuine
forced cross-type slot, 140 fell back to plain raw-SigLIP ranking because the
query's own type is unknown, and — importantly — **0 fell back for lack of any
cross-type candidate**. With 1,872 products spread across many fine-grained types,
a cross-type candidate was always findable; the "note this rather than silently
fill it" fallback path in the brief exists but was never actually needed here.

Full numbers for all three metric families (retrieval accuracy on both ground-truth
views, category diversity, popularity behavior) are in `results_table.md`. Headline
numbers:

| Configuration | Full HR@5 | Cross HR@5 | Diversity@5 | ARP@5 | Coverage@5 |
|---|---|---|---|---|---|
| Raw SigLIP (alone) | 0.502 | 0.076 | 0.330 | 42.78 | 0.918 |
| Phase 8 alone | 0.441 | 0.078 | 0.415 | 41.41 | 0.947 |
| Phase 9 alone | 0.315 | 0.046 | **0.579** | 39.52 | 0.950 |
| Blend: 0.5 raw + 0.5 Phase 8 | 0.503 | 0.086 | 0.362 | 42.09 | 0.935 |
| Blend: 0.7 raw + 0.3 Phase 8 | **0.510** | 0.085 | 0.338 | 43.04 | 0.924 |
| Diversity-forcing | 0.488 | **0.087** | 0.454 | 41.49 | 0.920 |

Three things stand out immediately:

1. **The phase 8 blends dominate the phase 9 blends on every metric.** At every
   matched alpha, blending with phase 8's compatibility signal gives equal or better
   accuracy (both views) and comparable or better diversity than blending with
   phase 9's — phase 9's transfer weakness (established in phases 9-10) carries
   straight through into any blend that uses it.
2. **A 0.5/0.5 blend with phase 8 is close to a free lunch on accuracy.** It ties
   raw SigLIP on full-view Hit Rate@5 (0.503 vs 0.502) while meaningfully improving
   cross-type Hit Rate@5 (0.086 vs 0.076, a 13% relative gain) — the first
   configuration in this project to improve the *targeted* cross-type metric without
   costing anything on the *aggregate* metric it's normally traded against.
3. **Phase 9 alone has, by a wide margin, the highest category diversity score of
   any configuration (0.579)** — even higher than the diversity-forcing
   configuration built specifically to increase diversity (0.454). Section
   "Explicit verdict" below addresses directly why this is not a point in phase 9's
   favor.

**A diversity-metric artifact worth flagging**: the diversity-forcing
configuration's Diversity@10 (0.410) is *lower* than its Diversity@5 (0.454) — the
only configuration where diversity drops going from K=5 to K=10. This is a direct,
mechanical consequence of the single-slot design described above: only rank 5 is
forced, so ranks 6-10 revert to plain raw-SigLIP ranking (which has its own,
lower, baseline diversity), diluting the average as K grows. This is not a bug, it's
what "one forced slot, not two" actually produces, and it is worth knowing before
reading Diversity@10 numbers for this configuration as if the whole list were being
diversified.

## Step 5: qualitative check — confirms and sharpens the numbers

`qualitative_examples/` has 4 five-configuration grids for the watch
(B005NGRC0W), laundry bag (B01FWDLMYC), Batgirl costume (B01B5BIU0E), and kurta
(B01AIT77RQ) queries. The best blend shown is 0.5 raw + 0.5 Phase 8 (picked for the
reasons in point 2 above — the strongest accuracy/diversity trade-off among the six
blends).

- **Laundry bag**: raw SigLIP and the blend both retrieve genuinely varied,
  practical bag/pouch-shaped items (shoe sleeves, a drawstring pouch, a poncho
  liner, a zippered case) — no color bias here even under raw SigLIP, since the
  shape signal dominates. Phase 9 alone is unambiguous: all five results are pink
  or bright-colored items with no functional connection to a laundry bag (a pink
  baby shoe, a pink-and-black zebra-print hat, a pink tank top, a pink sneaker) —
  the color-matching shortcut in its clearest form. The diversity-forcing
  configuration produced an *identical* top-5 to raw SigLIP for this query: raw
  SigLIP's own 5th-ranked item already happened to be a different fine-grained type,
  so the forced-slot rule had nothing to change. This is a real, useful negative
  case — diversity-forcing is a no-op whenever raw ranking is already diverse enough
  on its own.
- **Watch**: raw SigLIP, phase 8, and the blend all return other watches (genuine
  substitutes, not complements) — visually near-identical top-5s. Phase 9 alone is
  again dominated by watches, but its #4 result is a genuine complement (a watch
  winder/box), matching phase 9's original qualitative finding: the Polyvore signal
  is not *always* wrong, it just isn't reliable. The diversity-forcing
  configuration's forced slot is an AmazonBasics travel umbrella — technically a
  different type, but not a remotely plausible complement for a watch. This is the
  clearest illustration of diversity-forcing's core weakness: the rule only checks
  "different type + highest raw similarity," with no requirement that the result
  make any sense as a recommendation.
- **Batgirl costume**: raw SigLIP, phase 8, and the blend all retrieve genuine
  costume components (masks, capes, a matching Batman shirt, the Batgirl costume
  box) — the blend's top-5 is close to a reordering of phase 8's own list. Phase 9
  alone repeats phase 9/10's documented failure: black-and-yellow Batman-branded
  items (a wristwatch, boxer shorts) with no costume relationship. Diversity-forcing
  again matches raw SigLIP exactly here (rank 5 was already a mask, a different
  type) — a second no-op case.
- **Kurta**: raw SigLIP, phase 8, and the blend all retrieve other kurtas (genuine
  substitutes). Phase 9 alone is the worst failure of the four queries: results
  include a sizing-chart image, plain maroon polo shirts, and a Superman costume —
  color-matched to the query's dark red but with no garment-type or use-case
  relationship at all. Diversity-forcing's forced slot here is "Men's Cotton Stand
  Collar Shirt" — visually almost indistinguishable from the query's own kurtas.
  This is very likely an artifact of the exact limitation phase 8 already
  documented: the type-index-3 field for Traditional & Cultural Wear encodes
  *region*, not garment type (see phase 8's `category_level_check.md`), so two
  visually-identical kurta-style garments can carry different type labels for
  reasons unrelated to actual product category. This does not invalidate the
  aggregate diversity numbers (phase 8 already flagged and conservatively handled
  this same caveat, and it affects a bounded, documented subset of departments,
  not "a large fraction of the evaluation sample" — so the brief's fallback
  condition for reporting a blocking data problem was not triggered) but it does
  mean any single diversity-forcing example drawn from Traditional & Cultural Wear,
  Costumes & Accessories, or Baby should be read with that caveat in mind.

## Step 6: explicit verdict — is phase 9's color-matching shortcut harmless or harmful?

**Harmful, not harmless, now confirmed by full-sample measurement rather than a
couple of examples.** Phase 9 alone has the highest category-diversity score of any
configuration tested (0.579 at K=5, well above even the purpose-built
diversity-forcing configuration's 0.454) — read in isolation, that number would
suggest the color-matching shortcut is accidentally doing something useful for
variety. But the same configuration has the *worst* retrieval accuracy of any
configuration tested, on both ground-truth views (full Hit Rate@5 = 0.315, the
lowest of all ten configurations; cross-type Hit Rate@5 = 0.046, also the lowest).
And the qualitative grids show exactly why: phase 9's "diversity" comes from
retrieving items that are a different category than the query only because they
happen to share the query's dominant color — a pink laundry bag surfacing a pink
baby shoe, a Batgirl costume surfacing a Batman-branded wristwatch, a maroon kurta
surfacing a maroon polo shirt. These are technically cross-type, so they inflate the
diversity metric, but they are not real complements, and the accuracy numbers (built
from real `also_buy` co-purchase behavior, not a proxy) confirm that directly: real
customers are not buying these pairings together. A diversity metric that a
degenerate, low-accuracy signal can trivially maximize is not evidence the signal is
secretly good — it is evidence the diversity metric alone is gameable, and needs to
be read alongside accuracy, exactly as this phase's three-way evaluation design was
built to check.

## Recommendation

**Ship the 0.5/0.5 blend of raw SigLIP and phase 8's Amazon-trained compatibility
model** (`blended = 0.5 * raw_similarity + 0.5 * phase8_similarity`) as the primary
retrieval configuration, if any of the ten tested here were to move to production:

- It ties raw SigLIP on full-view Hit Rate@5 (0.503 vs. 0.502) and stays within
  0.005 on Hit Rate@10 (0.573 vs. 0.578) — no meaningful cost on the metric that
  matters most for overall retrieval quality.
- It improves cross-type Hit Rate@5 by 13% relative to raw SigLIP alone (0.086 vs.
  0.076) — a genuine, not illusory, improvement on the "does this actually
  recommend complements" question, confirmed qualitatively (the Batgirl and kurta
  grids show the blend keeping phase 8's genuine complements, not phase 9's color
  noise).
- It raises category diversity by about 10% relative to raw SigLIP (0.362 vs.
  0.330) without the accuracy collapse phase 9 alone shows.
- Popularity behavior and catalog coverage are essentially unchanged from raw
  SigLIP (ARP@5 42.09 vs. 42.78; coverage@5 0.935 vs. 0.918, slightly better) — the
  blend doesn't trade one problem (overspecialization) for another (popularity
  concentration).

If full-view accuracy is the only priority and even a small trade-off for diversity
is unwanted, **the 0.7/0.3 blend is a reasonable alternative** — it posts the single
best full-view Hit Rate@5 of any configuration tested (0.510), at the cost of most
of the diversity gain (0.338, barely above raw's 0.330).

**The diversity-forcing configuration is not recommended as a general-purpose
default**, despite posting the best cross-type Hit Rate@5 of all ten configurations
(0.087) and a real diversity lift (0.454 at K=5). The qualitative check shows its
forced slot is a blunt instrument: it is sometimes a no-op (raw SigLIP already had a
diverse item at rank 5, as in the laundry bag and Batgirl cases), sometimes
genuinely nonsensical (an umbrella for a watch query), and sometimes an artifact of
known type-taxonomy noise (the kurta case) rather than a real category difference.
It could be worth a narrower role — e.g., as an explicit "you might also consider"
slot presented separately from a primary ranked list, where a user can visually
judge a single suggestion on its own merits — but not as a blend into the main
ranking.

**Do not ship phase 9 alone, or any blend weighted toward phase 9, for Amazon
retrieval.** Its apparent diversity advantage is the color-matching shortcut
operating exactly as phases 9-10 diagnosed it, not a genuine complementary signal,
and it comes at the cost of the worst accuracy of any configuration tested on data
that reflects what customers actually buy together.

## On the brief's blocking-condition check

Category type information was not missing or unreliable for "a large fraction" of
the evaluation sample beyond what phase 8 already documented: 1,732/1,872 (92.5%)
of queries have a known type at index 3, identical to phase 8/9/10's figure, and the
diversity-forcing construction found a genuine cross-type candidate for every one of
those 1,732 queries (0 fallbacks for lack of a candidate). The one qualitative case
where the type field's reliability mattered (the kurta/Traditional & Cultural Wear
example above) is exactly the caveat phase 8 already flagged and handled
conservatively, not a new or larger problem — so this phase's diversity numbers can
be trusted at the same confidence level phase 8's cross-type numbers already carry.

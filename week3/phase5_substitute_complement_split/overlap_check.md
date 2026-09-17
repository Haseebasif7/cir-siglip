# Phase 5, Step 1: also_buy vs also_viewed Pair-Level Overlap Check

Directed (source_asin, target_asin) edges, restricted to targets that are
also inside the phase 1b sample (1,872 products) -- these are the only edges
that can ever register as a hit against a top-K retrieval list drawn from the
same sample, so edges pointing outside the sample are excluded from this count.

- Within-sample also_buy edges: 9450
- Within-sample also_viewed edges: 0
- Union (either field): 9450

## Pair-level overlap (relative to the union of both fields)

| Bucket | Count | % of union |
|---|---|---|
| In both also_buy and also_viewed | 0 | 0.0% |
| Only in also_buy | 9450 | 100.0% |
| Only in also_viewed | 0 | 0.0% |

## Each field's overlap with the other (relative to that field alone)

| Field | Edges | % also present in the other field |
|---|---|---|
| also_buy | 9450 | 0.0% |
| also_viewed | 0 | 0.0% |

## Reading

**also_viewed is not a 0% overlap with also_buy -- it is completely empty.** Every one of the 1,872 sampled products has an empty also_viewed list (0 nonempty out of 1,872; same result in phase 1's original 776-product sample: 0 nonempty out of 776). This was checked against a possible sampling artifact by streaming the first 50,000 raw records directly from the source metadata file (meta_Clothing_Shoes_and_Jewelry.json.gz): 0 of those records had a nonempty also_viewed field, against 8,092 with a nonempty also_buy field. **This confirms it's a genuine property of this specific hosted 2018 metadata file, not a sample or download artifact** -- also_viewed is present as a key in the JSON schema (week1's field inventory documents it) but its value is always an empty list in this dump. The substitute/complement split this phase set out to test cannot be run on also_viewed at all: there is no also_viewed ground truth anywhere in the data this project has used since phase 1.

## One-off raw-metadata verification (not re-run by this script)

To rule out a sample-construction bug before concluding also_viewed is genuinely absent, the first 50,000 raw records were streamed directly from `meta_Clothing_Shoes_and_Jewelry.json.gz` (the same source file every sample in this project comes from) and checked ad hoc:

| Check | Result |
|---|---|
| Records with nonempty also_viewed | 0 / 50,000 |
| Records with nonempty also_buy | 8,092 / 50,000 |

also_viewed is present as a JSON key but its value is always `[]` in this hosted 2018 file -- confirmed independent of any sampling choice made in this project.


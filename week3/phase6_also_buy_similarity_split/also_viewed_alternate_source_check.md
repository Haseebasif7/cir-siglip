# Phase 6, Step 5: Quick Check for an Alternate also_viewed Source

Bounded check, per the brief (not a deep investigation) -- stopped after the
searches below rather than trying to download and inspect any alternate copy.

## What was checked

1. **Web search** for the Amazon 2018 dataset's `also_viewed` field and any
   note of it being sparse/empty/deprecated. Result: no discussion found of
   this specific quirk. Search results mostly surfaced information about the
   *2014* dataset's schema (nested `related.also_viewed`/`related.also_bought`
   fields), which is a different, older schema than the flat `also_buy`/
   `also_viewed` fields used by the 2018 version this project relies on.
2. **Official docs pages** for the 2018 dataset:
   - `cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html` -- this is
     actually the *2014* dataset's docs page (an older, still-live URL); it
     documents `also_viewed` as part of the 2014 `related` object, not the
     2018 schema. Points to `amazon-reviews-2023.github.io` for the 2023
     version but has no comparative notes on the 2018 version.
   - `nijianmo.github.io/amazon/` -- this is the actual 2018 dataset docs
     page. It documents `also_viewed` as a flat field in the sample metadata
     (matching what this project has been parsing), but says nothing about
     it being sparse or empty for any category. No caveat is documented
     anywhere official about this.
3. **GitHub search** for any prior report of `also_viewed` being empty in the
   2018 dataset -- nothing found. This doesn't rule out the issue being known
   elsewhere; it just wasn't surfaced by this quick search.
4. **Kaggle mirrors** -- found two candidate re-hosted copies
   (`kaggle.com/datasets/rogate16/amazon-reviews-2018-full-dataset` and
   `kaggle.com/datasets/abhivish/amazon-fashion-review-dataset`). Could not
   inspect their actual field contents: Kaggle's dataset pages are
   JS-rendered and returned no usable content via a plain fetch, and
   downloading either (likely multi-GB) to inspect a single field was judged
   out of scope for a "quick, bounded" check. Both are very likely re-hosts
   of the same McAuley Lab source data this project already uses, not an
   independent scrape, so they would plausibly have the same empty
   `also_viewed` field -- but this is an inference, not a verified fact.

## Conclusion

**No alternate source with a confirmed-populated also_viewed field was
found in this quick check.** The search did clarify one useful thing: the
2014 version of this dataset (different schema, nested `related` object)
did have `also_viewed` populated with real data -- so the emptiness is
specific to this project's 2018 metadata source, not a property of "Amazon
relatedness data" in general. If a populated also_viewed signal is still
wanted later, the next concrete step (not attempted here, since it's out of
this phase's bounded scope) would be to download a small slice of a Kaggle
mirror or the 2014 dataset directly and check a sample of records for
non-empty also_viewed before committing to a full re-download.

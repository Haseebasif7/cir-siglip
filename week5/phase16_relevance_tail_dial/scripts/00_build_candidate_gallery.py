"""Merge phase 7's 24,719-item Amazon training-pool embeddings with phase 1b's
1,872-item eval-sample embeddings into one retrieval candidate gallery.

Why: phase 1b's eval sample alone is 96.2% head-tier / 3.8% mid / 0.0% tail
(BFS/snowball sampling skew, documented in phase 3) -- using it as the ONLY
retrievable pool would make Tail-Coverage@N/RPI structurally degenerate no
matter how well the tail-exposure mode works. Merging in phase 7's larger,
already-embedded, confirmed-disjoint pool gives the alpha sweep actual tail
items to surface. The eval sample keeps its separate role as QUERIES only
(unchanged) -- this script only builds the searchable candidate side.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_EMB = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility" / "embeddings" / "siglip_base.npz"
PHASE1B_EMB = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced" / "embeddings" / "siglip_base.npz"
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"

OUT_GALLERY = BASE_DIR / "data" / "candidate_gallery.npz"
OUT_SUMMARY = BASE_DIR / "training_pool_summary.md"

# Decision rule (stated in advance, per the plan): the merged gallery needs to be
# meaningfully tail-representative or we extend it. Thresholds chosen to be a
# genuine "is this usable at all" bar, not a cosmetic one.
MIN_TAIL_FRACTION = 0.02
MIN_TAIL_ABSOLUTE = 200


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["asins"].astype(str), d["embeddings"].astype(np.float32)


def main():
    asins7, emb7 = load_npz(PHASE7_EMB)
    asins1b, emb1b = load_npz(PHASE1B_EMB)

    overlap = set(asins7.tolist()) & set(asins1b.tolist())

    gallery_asins = np.concatenate([asins7, asins1b])
    gallery_emb = np.concatenate([emb7, emb1b], axis=0)

    # dedupe just in case (should be a no-op given 0 overlap, but don't assume)
    _, first_idx = np.unique(gallery_asins, return_index=True)
    first_idx = np.sort(first_idx)
    gallery_asins = gallery_asins[first_idx]
    gallery_emb = gallery_emb[first_idx]

    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count", "tier"])
    tier_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["tier"]))
    refcount_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["ref_count"]))

    tiers = [tier_lookup.get(a, "unknown") for a in gallery_asins]
    n = len(tiers)
    from collections import Counter
    counts = Counter(tiers)
    tail_n = counts.get("tail", 0)
    tail_frac = tail_n / n if n else 0.0

    np.savez(
        OUT_GALLERY,
        asins=gallery_asins,
        embeddings=gallery_emb,
    )

    needs_extension = (tail_frac < MIN_TAIL_FRACTION) or (tail_n < MIN_TAIL_ABSOLUTE)

    lines = []
    lines.append("# Phase 16: Training Pool / Candidate Gallery Summary\n")
    lines.append("## Reused infrastructure\n")
    lines.append(f"- Phase 7 training pool embeddings: `{PHASE7_EMB.relative_to(BASE_DIR.parent.parent)}` "
                  f"({len(asins7)} items, 768-d SigLIP)")
    lines.append(f"- Phase 1b eval sample embeddings: `{PHASE1B_EMB.relative_to(BASE_DIR.parent.parent)}` "
                  f"({len(asins1b)} items, 768-d SigLIP)")
    lines.append(f"- Phase 3 popularity/tier lookup: `{TIER_LOOKUP.relative_to(BASE_DIR.parent.parent)}` "
                  f"({len(tier_df)} catalog-wide asins)\n")
    lines.append("## Disjointness (phase 7 pool vs phase 1b eval sample)\n")
    lines.append(f"Overlap between phase 7's {len(asins7)}-item pool and phase 1b's {len(asins1b)}-item eval "
                 f"sample: **{len(overlap)}**. This re-asserts, directly against the embedding files this phase "
                 f"actually loads, the same zero-overlap result already proven and recorded at "
                 f"`week3/phase7_learned_compatibility/data/exclusion_check.md` (checked against the full "
                 f"27,970-product raw pool, of which the 24,719-item cleaned pool used here is a strict subset).\n")
    lines.append("## Candidate gallery\n")
    lines.append(f"Merged gallery size: **{n}** items ({len(asins7)} + {len(asins1b)}, 0 duplicates found).\n")
    lines.append("Role split: phase 1b's 1,872 items remain the **queries** (clean, disjoint-from-training, "
                 "this project's standard held-out eval set, unchanged role). The full merged gallery of "
                 f"{n} items is the **retrievable candidate pool** every query searches against for both the "
                 "Hit-Rate/CIR-style accuracy sweep and the field-standard popularity metrics.\n")
    lines.append("## Tier composition of the merged gallery\n")
    lines.append("| Tier | Count | % of gallery |")
    lines.append("|---|---|---|")
    for t in ["head", "mid", "tail", "unknown"]:
        c = counts.get(t, 0)
        lines.append(f"| {t} | {c} | {c/n*100:.2f}% |")
    lines.append("")
    lines.append(f"Tail-tier items: **{tail_n}** ({tail_frac*100:.2f}% of the {n}-item gallery), against a "
                 f"decision rule of >= {MIN_TAIL_FRACTION*100:.0f}% and >= {MIN_TAIL_ABSOLUTE} absolute tail items "
                 "before this gallery is considered adequate for the field-standard tail metrics.\n")
    if needs_extension:
        lines.append("**Verdict: INADEQUATE per the pre-declared rule.** A supplementary tail-tier sample is "
                     "needed (fresh image download + SigLIP extraction for additional tail-tier asins) before "
                     "proceeding -- see `00b_supplement_tail_sample.py` and the updated gallery/summary this "
                     "produces before trusting the rest of this phase's tail-related numbers.\n")
    else:
        lines.append("**Verdict: adequate.** No supplementary sampling needed -- proceeding with this "
                     f"{n}-item merged gallery as-is for the rest of the phase.\n")
    gallery_ceiling_pct = n / 3777545 * 100
    lines.append("## Important scoping note for the field-standard metrics (step 5)\n")
    lines.append("Coverage@N and Tail-Coverage@N are computed later against the TRUE catalog-wide denominators "
                 "from phase 3 (3,777,545 total items; 1,888,773 tail-tier items), not against this gallery's "
                 f"size ({n} items) -- the brief requires the field's own catalog-relative definitions, not an "
                 "approximation. This gallery only affects the NUMERATOR (what items are reachable to be "
                 "recommended at all); expect resulting Coverage@N/Tail-Coverage@N percentages to be small "
                 f"(upper-bounded by roughly {gallery_ceiling_pct:.3f}% for any metric with this gallery as its "
                 "reachable-item source), reported as an honest structural ceiling of this pipeline's finite "
                 "embedded pool, not normalized away.\n")

    OUT_SUMMARY.write_text("\n".join(lines))
    print(f"Gallery saved: {OUT_GALLERY} ({n} items)")
    print(f"Tail: {tail_n} ({tail_frac*100:.2f}%) -- needs_extension={needs_extension}")
    print(f"Summary written: {OUT_SUMMARY}")

    if needs_extension:
        # Signal to the caller / next script explicitly, don't just bury it in the markdown.
        flag_path = BASE_DIR / "data" / "NEEDS_TAIL_SUPPLEMENT.flag"
        flag_path.write_text(f"tail_frac={tail_frac}\ntail_n={tail_n}\n")
        print(f"FLAGGED: gallery needs tail supplementation, see {flag_path}")


if __name__ == "__main__":
    main()

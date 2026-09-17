"""
Phase 3, Step 3: full per-query top-10 retrieval lists.

Reuses phase 1b's already-computed retrieval_results.json rather than
recomputing from embeddings -- that file already contains, per technique,
per query, the top-10 retrieved asins by cosine similarity over the exact
1,872-product category-balanced sample this phase also uses. Only the
fields phase 3 needs (asin, category, retrieved) are kept; phase 1b's own
hit-rate bookkeeping fields are dropped since they belong to that phase's
own report, not this one.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1B_RETRIEVAL = BASE_DIR.parent / "phase1b_category_balanced" / "data" / "retrieval_results.json"
OUT_PATH = BASE_DIR / "data" / "full_retrievals.json"

TECHNIQUES = ["resnet50", "clip_vit_b32", "fashionclip", "siglip_base"]


def main():
    with open(PHASE1B_RETRIEVAL) as f:
        src = json.load(f)

    out = {}
    for technique in TECHNIQUES:
        if technique not in src:
            print(f"WARNING: {technique} missing from {PHASE1B_RETRIEVAL}")
            continue
        out[technique] = [
            {
                "asin": entry["asin"],
                "category": entry["category"],
                "retrieved_top10": entry["retrieved"],
            }
            for entry in src[technique]
        ]
        print(f"{technique}: {len(out[technique])} queries")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

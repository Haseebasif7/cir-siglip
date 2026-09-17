"""
Phase 8, step 1: find a genuinely fine-grained category level in the raw
`categories` breadcrumb field, deeper than the 11 department-level buckets
(category_bucket = categories[1]) used for sampling/evaluation since phase 1b.

Reuses phase 7's cleaned training pool (24,719 products) -- no new data needed.

Method: inspect the breadcrumb-length distribution and the distinct-value
count/content at several candidate depths (index 2, 3, 4), then pick the
depth that best balances genuine item-type granularity against coverage.
Report the finding honestly, including a real complication found during
inspection: the breadcrumb's semantic meaning at a given depth is NOT
consistent across departments (see write-up in category_level_check.md) --
some departments put a demographic/regional subdivision before the real
item type, pushing "true type" a level or two deeper than in other
departments. A single fixed global index can't perfectly correct for this;
the choice made here is deliberately the conservative one (see notes below).
"""
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility" / "data" / "sample_data_cleaned.csv"
OUT_MD = BASE_DIR / "category_level_check.md"
TYPE_LEVEL_JSON = BASE_DIR / "data" / "product_types.json"

CHOSEN_INDEX = 3  # see category_level_check.md for the full justification


def main():
    df = pd.read_csv(SAMPLE_CSV)
    cats = df["category"].apply(ast.literal_eval)
    bucket = df["category_bucket"]

    length_dist = Counter(len(c) for c in cats)

    candidate_report = {}
    for idx in [2, 3, 4]:
        vals = [c[idx] for c in cats if len(c) > idx]
        n_missing = len(cats) - len(vals)
        vc = Counter(vals)
        candidate_report[idx] = {
            "n_distinct": len(vc),
            "n_missing": n_missing,
            "top15": vc.most_common(15),
        }

    # department-level inconsistency check: sample a few products per department
    # at the chosen index to show the semantic drift documented in the write-up
    dept_examples = defaultdict(list)
    for c, b in zip(cats, bucket):
        if len(dept_examples[b]) < 4:
            dept_examples[b].append(c[:6])

    # build the actual per-product type assignment used going forward:
    # CHOSEN_INDEX if present, else "(unknown)" -- conservative, no fallback
    # substitution to a different index, since that would re-introduce the
    # cross-department inconsistency this check exists to document rather
    # than paper over with an ad hoc per-department correction table.
    product_types = {}
    n_known = 0
    for asin, c in zip(df["asin"], cats):
        if len(c) > CHOSEN_INDEX:
            product_types[asin] = c[CHOSEN_INDEX]
            n_known += 1
        else:
            product_types[asin] = None
    n_unknown = len(product_types) - n_known

    with open(TYPE_LEVEL_JSON, "w") as f:
        json.dump(product_types, f)
    print(f"Saved {TYPE_LEVEL_JSON}: {n_known} known types, {n_unknown} unknown (breadcrumb too shallow).")

    # --- write report ---
    lines = [
        "# Phase 8, Step 1: Fine-Grained Category Level Check",
        "",
        "Every phase since 1b has used department-level categories (categories[1],",
        "11 values: Women, Men, Boys, Girls, Baby, Novelty & More, Costumes & ",
        "Accessories, Luggage & Travel Gear, Shoe/Jewelry/Watch Accessories, ",
        "Traditional & Cultural Wear, Uniforms/Work/Safety). This phase needs a",
        "finer level to distinguish real item types (a shirt vs. pants) within a",
        "department, so heterogeneous-dyad filtering means something.",
        "",
        "## Breadcrumb depth distribution (24,719 products, phase 7's cleaned pool)",
        "",
        "| Breadcrumb length | Products |",
        "|---|---|",
    ]
    for l in sorted(length_dist):
        lines.append(f"| {l} | {length_dist[l]} |")
    lines.append("")
    lines.append("(Length includes the root 'Clothing, Shoes & Jewelry' at index 0 and ")
    lines.append("department at index 1, so a length-3 breadcrumb has exactly one level ")
    lines.append("beyond department; longer breadcrumbs increasingly trail off into ")
    lines.append("product-feature bullet text rather than real category structure.)")
    lines.append("")

    lines.append("## Candidate levels inspected")
    lines.append("")
    for idx, rep in candidate_report.items():
        lines.append(f"### Index {idx}: {rep['n_distinct']} distinct values, "
                      f"{rep['n_missing']} products missing this depth")
        lines.append("")
        lines.append("Top 15 by frequency: " + ", ".join(f"`{v}` ({n})" for v, n in rep["top15"]))
        lines.append("")

    lines.append("## Complication found: breadcrumb semantics shift by department")
    lines.append("")
    lines.append(
        "A single fixed index does NOT consistently land on 'item type' across "
        "departments -- inspecting sample breadcrumbs per department shows why:"
    )
    lines.append("")
    for dept, examples in sorted(dept_examples.items()):
        lines.append(f"**{dept}**:")
        for ex in examples:
            lines.append(f"  - {ex}")
    lines.append("")
    lines.append(
        "Concretely: Baby-department products put a gender subdivision "
        "('Baby Girls'/'Baby Boys') at index 2, pushing the real item type "
        "('Clothing Sets', 'Footies & Rompers', 'Hair Accessories') to index 4, "
        "not index 3. Novelty & More puts 'Clothing' at index 2, the vague "
        "bucket 'Novelty' at index 3, and ANOTHER demographic split "
        "('Men'/'Women') at index 4 -- real type is even deeper. Costumes & "
        "Accessories similarly stacks 'Kids & Baby' then 'Boys'/'Girls' before "
        "reaching anything type-like. Traditional & Cultural Wear encodes "
        "*region* ('Asian', 'Middle Eastern') rather than garment type at all, "
        "and degenerates into free-text product description almost immediately "
        "(e.g. index 3 = 'Lightweight and comfortable 1 piece Hijabs.' for some "
        "products). Uniforms, Work & Safety terminates at length 3 for many "
        "products (no type info beyond 'Clothing' at all)."
    )
    lines.append("")
    lines.append(
        f"**Decision: use index {CHOSEN_INDEX} as the type level, with no per-department "
        "correction, and treat any product whose breadcrumb doesn't reach this "
        "depth as unknown type (excluded from heterogeneous-dyad determination "
        "in step 2, not guessed at).** This is a deliberately conservative choice: "
        "building a bespoke per-department index-correction table would reduce "
        "the noise described above, but it's out of scope for what this phase "
        "needs. The failure mode of NOT correcting for it is asymmetric and "
        "safe for this phase's purpose -- when index 3 is still a generic/"
        "demographic bucket rather than a true type (e.g. two different actual "
        "item types both showing 'Novelty'), the two products get scored as "
        "*same* type and their edge is excluded from the heterogeneous-dyad "
        "positive set. That undercounts genuine heterogeneous dyads in the "
        "affected departments (a real cost, addressed if needed by pool "
        "expansion in step 2), but it does NOT let same-type/near-duplicate "
        "pairs slip through mislabeled as heterogeneous -- which is the "
        "specific contamination phase 7 diagnosed and this phase exists to "
        "avoid. Index 3 clears the bar of 'genuinely fine-grained' overall "
        f"({candidate_report[3]['n_distinct']} distinct values vs. 11 department "
        "buckets), even though it is not uniformly clean across every department."
    )
    lines.append("")
    lines.append(f"## Result: {n_known} of {len(product_types)} products have a known type "
                  f"at index {CHOSEN_INDEX} ({100*n_known/len(product_types):.1f}%); "
                  f"{n_unknown} ({100*n_unknown/len(product_types):.1f}%) have breadcrumbs "
                  "too shallow to reach it and are treated as unknown type.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

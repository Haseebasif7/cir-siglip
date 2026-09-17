"""
Phase 5, step 2: re-score the already-computed top-5/top-10 retrieval lists
(SigLIP and FashionCLIP only, phase 1b's exact 1,872-product sample) against
also_viewed-only, also_buy-only, and the combined signal separately.

No new embeddings, no recomputation of similarity -- reuses phase 1b's
`data/retrieval_results.json`, which already stores each query's top-10
retrieved asins verbatim. Ground truth for each of the three variants comes
from step 1's `data/signal_edges.json`.

Step 1 already found also_viewed is empty for every product in this sample
(and in phase 1's original sample, and in a 50,000-record raw-metadata check)
-- so the also_viewed-only numbers here are expected to be trivially 0.000
for every query and every technique. They're still computed and reported
explicitly, per the phase brief, rather than skipped.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RETRIEVAL_JSON = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced" / "data" / "retrieval_results.json"
SIGNAL_EDGES_JSON = BASE_DIR / "data" / "signal_edges.json"
OUT_MD = BASE_DIR / "results_table.md"

TECHNIQUES = ["siglip_base", "fashionclip"]
TECHNIQUE_LABELS = {"siglip_base": "SigLIP", "fashionclip": "FashionCLIP"}
K_VALUES = [5, 10]
MIN_OVERLAP_REFS_TO_TRUST = 20  # same threshold phase 1b used


def load_signal_edges():
    with open(SIGNAL_EDGES_JSON) as f:
        d = json.load(f)
    also_buy = {k: set(v) for k, v in d["also_buy"].items()}
    also_viewed = {k: set(v) for k, v in d["also_viewed"].items()}
    return also_buy, also_viewed


def score(per_query, ground_truth):
    """ground_truth: dict asin -> set of related asins (already filtered to
    whatever signal is being scored). Returns hit_rate@5, hit_rate@10,
    precision@5, precision@10, plus the number of within-sample overlap refs
    this ground truth actually has (for the trustworthy-threshold check)."""
    n = len(per_query)
    hits = {k: 0 for k in K_VALUES}
    prec_sum = {k: 0.0 for k in K_VALUES}
    n_overlap_refs = 0

    for q in per_query:
        asin = q["asin"]
        related = ground_truth.get(asin, set())
        retrieved_all = q["retrieved"]
        for k in K_VALUES:
            retrieved_k = retrieved_all[:k]
            n_hits = sum(1 for r in retrieved_k if r in related)
            if n_hits > 0:
                hits[k] += 1
            prec_sum[k] += n_hits / k

    # within-sample overlap refs for this signal = count of (asin, target) edges
    # where target is also a query in this per_query list (i.e. inside the sample)
    sample_asins = {q["asin"] for q in per_query}
    for asin, related in ground_truth.items():
        if asin in sample_asins:
            n_overlap_refs += len(related & sample_asins)

    metrics = {f"hit_rate@{k}": hits[k] / n for k in K_VALUES}
    metrics.update({f"precision@{k}": prec_sum[k] / n for k in K_VALUES})
    metrics["n_overlap_refs"] = n_overlap_refs
    metrics["trustworthy"] = n_overlap_refs >= MIN_OVERLAP_REFS_TO_TRUST
    return metrics


def main():
    also_buy, also_viewed = load_signal_edges()
    combined = {asin: also_buy.get(asin, set()) | also_viewed.get(asin, set())
                for asin in set(also_buy) | set(also_viewed)}

    with open(RETRIEVAL_JSON) as f:
        all_retrieval = json.load(f)

    results = {}
    for tech in TECHNIQUES:
        per_query = all_retrieval[tech]
        results[tech] = {
            "also_viewed_only": score(per_query, also_viewed),
            "also_buy_only": score(per_query, also_buy),
            "combined": score(per_query, combined),
        }

    lines = [
        "# Phase 5, Step 2: Hit Rate@K by Ground-Truth Signal (SigLIP and FashionCLIP)",
        "",
        "Same 1,872-product sample and same top-10 retrieved lists as phase 1b "
        "(no recomputation) -- only the ground truth used for scoring changes per row.",
        "",
        "| Technique | Signal | Overlap refs (within sample) | Trustworthy (>=20 refs)? "
        "| Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tech in TECHNIQUES:
        label = TECHNIQUE_LABELS[tech]
        for signal_key, signal_label in [
            ("also_viewed_only", "also_viewed only"),
            ("also_buy_only", "also_buy only"),
            ("combined", "combined (reference)"),
        ]:
            m = results[tech][signal_key]
            trust = "yes" if m["trustworthy"] else "**NO -- too few refs**"
            hr5 = f"{m['hit_rate@5']:.3f}" if m["trustworthy"] or signal_key == "combined" else "n/a"
            hr10 = f"{m['hit_rate@10']:.3f}" if m["trustworthy"] or signal_key == "combined" else "n/a"
            p5 = f"{m['precision@5']:.3f}" if m["trustworthy"] or signal_key == "combined" else "n/a"
            p10 = f"{m['precision@10']:.3f}" if m["trustworthy"] or signal_key == "combined" else "n/a"
            lines.append(
                f"| {label} | {signal_label} | {m['n_overlap_refs']} | {trust} "
                f"| {hr5} | {hr10} | {p5} | {p10} |"
            )

    lines.append("")
    lines.append(
        "**also_viewed_only is flagged not trustworthy for both techniques (0 overlap "
        "refs, far below the 20-ref threshold) -- there is no also_viewed ground truth "
        "anywhere in this sample (see overlap_check.md), so Hit Rate@K against it is "
        "0.000/0.000 by construction (no query ever has a nonempty also_viewed ground "
        "truth to hit), not a genuine measurement of retrieval quality against a "
        "substitute-like signal.**"
    )
    lines.append("")
    lines.append(
        "also_buy_only is numerically identical to combined for both techniques -- "
        "expected, since combined = also_buy UNION also_viewed, and also_viewed "
        "contributes the empty set for every product in this sample."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    for tech in TECHNIQUES:
        print(tech, results[tech])

    # Save raw json too, in case phase 5 notes/report need to reference exact numbers later
    with open(BASE_DIR / "data" / "split_signal_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

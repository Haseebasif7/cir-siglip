"""Field-standard long-tail metrics, identical methodology to phases
16/16c (same catalog-wide denominators, same N values), for direct
comparability.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"
CATALOG_TOTAL = 3_777_545
CATALOG_TAIL_TOTAL = 1_888_773

CACHE = BASE_DIR / "data" / "alpha_sweep_retrievals.npz"
OUT_RESULTS_MD = BASE_DIR / "results_table.md"

N_VALUES = [10, 20]

PHASE16_REFERENCE = {
    10: {0.0: {"apri": 16.58, "rpi": 0.5230}, 1.0: {"apri": 17.21, "rpi": 0.5233}},
    20: {0.0: {"apri": 16.23, "rpi": 0.5149}, 1.0: {"apri": 16.92, "rpi": 0.5171}},
}
PHASE16C_REFERENCE = {
    10: {0.0: {"apri": 17.39, "rpi": 0.5072}, 1.0: {"apri": 17.27, "rpi": 0.4972}},
    20: {0.0: {"apri": 17.19, "rpi": 0.5015}, 1.0: {"apri": 17.39, "rpi": 0.4920}},
}


def main():
    d = np.load(CACHE, allow_pickle=True)
    alphas = d["alphas"]
    retrieved_asins = d["retrieved_asins"]

    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count", "tier"])
    refcount_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["ref_count"].astype(int)))
    tier_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["tier"]))

    rows = {N: [] for N in N_VALUES}
    for ai, alpha in enumerate(alphas):
        for N in N_VALUES:
            topN = retrieved_asins[ai, :, :N]
            flat = topN.reshape(-1)

            unique_items = set(flat.tolist())
            coverage = len(unique_items) / CATALOG_TOTAL
            tail_items_seen = {a for a in unique_items if tier_lookup.get(a) == "tail"}
            tail_coverage = len(tail_items_seen) / CATALOG_TAIL_TOTAL

            refcounts = np.array([refcount_lookup.get(a, 0) for a in flat], dtype=np.float64)
            apri = float(refcounts.mean())
            is_head = np.array([tier_lookup.get(a) == "head" for a in flat])
            rpi = float(is_head.mean())

            rows[N].append({
                "alpha": float(alpha), "coverage": coverage, "tail_coverage": tail_coverage,
                "apri": apri, "rpi": rpi, "n_unique": len(unique_items), "n_unique_tail": len(tail_items_seen),
            })
            print(f"N={N} alpha={alpha:.1f}: coverage={coverage:.6f} tail_coverage={tail_coverage:.6f} "
                  f"apri={apri:.2f} rpi={rpi:.4f}")

    lines = ["\n## Field-standard long-tail metrics\n"]
    lines.append(f"Same formulas and catalog-wide denominators as phases 16/16c ({CATALOG_TOTAL:,} total items, "
                 f"{CATALOG_TAIL_TOTAL:,} tail-tier items).\n")

    for N in N_VALUES:
        lines.append(f"### N={N}\n")
        lines.append("| alpha | Coverage@N | Tail-Coverage@N | APRI | RPI |")
        lines.append("|---|---|---|---|---|")
        for r in rows[N]:
            lines.append(f"| {r['alpha']:.1f} | {r['coverage']*100:.4f}% ({r['n_unique']}) | "
                         f"{r['tail_coverage']*100:.4f}% ({r['n_unique_tail']}) | {r['apri']:.2f} | {r['rpi']:.4f} |")
        lines.append("")

    lines.append("### Direct three-way comparison (APRI / RPI at endpoints)\n")
    for N in N_VALUES:
        r1 = next(r for r in rows[N] if r["alpha"] == 1.0)
        r0 = next(r for r in rows[N] if r["alpha"] == 0.0)
        p16_0, p16_1 = PHASE16_REFERENCE[N][0.0], PHASE16_REFERENCE[N][1.0]
        p16c_0, p16c_1 = PHASE16C_REFERENCE[N][0.0], PHASE16C_REFERENCE[N][1.0]
        lines.append(f"- N={N}: **Phase 16** APRI {p16_0['apri']:.2f}(tail)/{p16_1['apri']:.2f}(rel), RPI "
                     f"{p16_0['rpi']:.4f}/{p16_1['rpi']:.4f}. **Phase 16c** APRI {p16c_0['apri']:.2f}/"
                     f"{p16c_1['apri']:.2f}, RPI {p16c_0['rpi']:.4f}/{p16c_1['rpi']:.4f}. **Phase 16d** APRI "
                     f"{r0['apri']:.2f}/{r1['apri']:.2f}, RPI {r0['rpi']:.4f}/{r1['rpi']:.4f}.")
    lines.append("")

    with open(OUT_RESULTS_MD, "a") as f:
        f.write("\n".join(lines))
    print(f"Appended field-standard metrics to {OUT_RESULTS_MD}")


if __name__ == "__main__":
    main()

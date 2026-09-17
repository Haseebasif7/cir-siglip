"""
Phase 13, step 5: assemble results_table.md -- the CSA-Net reproduction's
Recall@10/30/50 under this project's own CIR harness, alongside CSA-Net's
own published numbers (paper Table 4, Polyvore Outfits non-disjoint) and
every configuration already evaluated on this exact harness in phases 12,
12b, 12c (numbers copied from those phases' own results_table.md files,
not re-run -- the benchmark file is identical/unchanged across all of them).
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE_DIR / "data" / "csa_cir_results.json"
OUT_MD = BASE_DIR / "results_table.md"

# Published CSA-Net numbers, Table 4, "Polyvore Outfits" (non-disjoint) column
# -- the split this project's own harness also uses.
PUBLISHED = {"10": 0.0827, "30": 0.1567, "50": 0.2091}

# Copied verbatim from week4/phase12c_ranking_distillation/results_table.md
# (same unchanged benchmark: 26494 pool slots, 29681 queries).
PROJECT_CONFIGS = [
    ("Raw SigLIP (alone)", 0.0553, 0.1067, 0.1437),
    ("Phase 9 Model A (alone)", 0.1317, 0.2464, 0.3216),
    ("Phase 12c: Substitute mode", 0.0667, 0.1307, 0.1734),
    ("Phase 12c: Complement mode", 0.0971, 0.1875, 0.2471),
    ("Phase 12c: Blend (0.5)", 0.0893, 0.1695, 0.2255),
]


def main():
    with open(RESULTS_JSON) as f:
        result = json.load(f)
    recall = result["recall"]
    n_total = result["n_total"]
    n_skipped = result["n_skipped"]

    lines = [
        "# Phase 13: CSA-Net Reproduction -- Results Table",
        "",
        f"Evaluated on this project's own CIR harness (`week4/phase12_controllable_modes/"
        f"data/cir_benchmark.json`, unchanged since phase 12): 26494 pool slots, 29681 queries "
        f"total in the benchmark; {n_total} scored for CSA-Net "
        f"({n_skipped} skipped -- see `phase13_notes.md` for why, if nonzero).",
        "",
        "## This project's reproduction vs. CSA-Net's own published numbers",
        "",
        "| | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| **This reproduction (own harness)** | {recall['10']:.4f} | {recall['30']:.4f} | {recall['50']:.4f} |",
        f"| CSA-Net published (paper Table 4, Polyvore Outfits non-disjoint) | {PUBLISHED['10']:.4f} | {PUBLISHED['30']:.4f} | {PUBLISHED['50']:.4f} |",
        "",
        "Note: the published numbers come from the paper's OWN candidate-pool construction "
        "(27/153 fine-grained categories, each capped at 3,000 images) which differs from this "
        "project's own harness (11 broad semantic categories, same 3,000 cap) -- so this is a "
        "similar-protocol sanity check, not a byte-for-byte identical evaluation. See "
        "`phase13_notes.md` for what a close/far match does and doesn't establish.",
        "",
        "## Against this project's own configurations (identical harness, identical benchmark)",
        "",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
    ]
    for name, r10, r30, r50 in PROJECT_CONFIGS:
        lines.append(f"| {name} | {r10:.4f} | {r30:.4f} | {r50:.4f} |")
    lines.append(f"| **Phase 13: CSA-Net reproduction** | **{recall['10']:.4f}** | "
                  f"**{recall['30']:.4f}** | **{recall['50']:.4f}** |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()

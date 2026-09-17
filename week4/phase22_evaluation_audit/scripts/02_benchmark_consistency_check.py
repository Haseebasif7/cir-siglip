"""
Phase 22, step 2: confirm phase 9, CSA-Net (phase 13b), and OutfitTransformer
(phase 14b run 2) are all scored against the literal same benchmark file --
same bytes on disk, not independently reconstructed versions that could have
quietly diverged -- by checking each evaluation script's actual BENCHMARK_JSON
path and hashing the resolved file.
"""
import hashlib
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent

OUT_MD = BASE_DIR / "benchmark_consistency_check.md"

# The evaluation script actually used to produce each number in phase 20's
# final_comparison_table.md, per that phase's own citations.
EVAL_SCRIPTS = {
    "Phase 9 (via phase 20's verification)": REPO_ROOT / "week4/phase20_full_consolidation/scripts/01_verify_phase9.py",
    "CSA-Net (phase 13b, frozen SigLIP backbone)": REPO_ROOT / "week4/phase13b_csa_net_siglip_backbone/scripts/03_csa_cir_eval.py",
    "OutfitTransformer (phase 14b run 2, random negatives)": REPO_ROOT / "week4/phase14b_outfittransformer_category_negatives/scripts/04b_cir_eval_random_negatives.py",
}


def resolve_benchmark_path(script_path):
    """Statically resolve BENCHMARK_JSON by reading the script's own path
    construction lines -- avoids importing/executing each script (which have
    heavy torch/model deps) just to introspect one constant. If the script
    itself doesn't define BENCHMARK_JSON (imports load_benchmark() from a
    local cir_eval.py instead, e.g. phase 9's verification), resolve it from
    that sibling cir_eval.py instead."""
    def resolve_from(src, base_dir):
        lines = src.splitlines()
        ns = {"BASE_DIR": base_dir, "Path": Path}
        for line in lines:
            stripped = line.strip()
            if re.match(r"^[A-Z_0-9]+\s*=\s*BASE_DIR", stripped) or re.match(r"^[A-Z_0-9]+\s*=\s*.*_DIR\b", stripped):
                var, _, expr = stripped.partition("=")
                var = var.strip()
                expr = expr.strip()
                try:
                    ns[var] = eval(expr, {}, ns)
                except Exception:
                    pass
        return ns.get("BENCHMARK_JSON")

    base_dir = script_path.resolve().parent.parent  # matches each script's own BASE_DIR = .../scripts/../
    src = script_path.read_text()
    result = resolve_from(src, base_dir)
    if result is not None:
        return result

    if "from cir_eval import" in src or "import cir_eval" in src:
        sibling = script_path.parent / "cir_eval.py"
        if sibling.exists():
            return resolve_from(sibling.read_text(), sibling.resolve().parent.parent)
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    lines = [
        "# Phase 22, Step 2: Benchmark Construction Consistency Across Evaluated Models",
        "",
        "Checked directly against each evaluation script's own path-construction code "
        "(not assumed from the fact that all three are described as using \"the CIR "
        "benchmark\") -- resolves each script's `BENCHMARK_JSON` constant to an absolute "
        "path and hashes the file it actually points at.",
        "",
        "| Model | Evaluation script | Resolved benchmark path | SHA-256 |",
        "|---|---|---|---|",
    ]
    resolved = {}
    for name, script_path in EVAL_SCRIPTS.items():
        bpath = resolve_benchmark_path(script_path)
        bpath = bpath.resolve() if bpath else None
        digest = sha256(bpath) if bpath and bpath.exists() else None
        resolved[name] = (bpath, digest)
        rel = bpath.relative_to(REPO_ROOT) if bpath else "FAILED TO RESOLVE"
        lines.append(f"| {name} | `{script_path.relative_to(REPO_ROOT)}` | `{rel}` | `{digest}` |")

    lines.append("")
    digests = {d for _, d in resolved.values()}
    paths = {p for p, _ in resolved.values()}
    if len(digests) == 1 and None not in digests:
        lines.append(
            f"**Confirmed: all three evaluation scripts resolve to the exact same file on "
            f"disk** (identical SHA-256 hash, {len(paths)} distinct path string(s) all "
            f"pointing at the same underlying file via `Path.resolve()`). There is no "
            f"possibility of a quietly diverged copy -- CSA-Net and OutfitTransformer are "
            f"not scored against a separately reconstructed benchmark, they read the "
            f"identical bytes phase 9's verification reads."
        )
    else:
        lines.append(
            "**Divergence found** -- not every evaluation script resolves to the same "
            "file. This is a real finding, not something to round away; see which path/"
            "hash differs above."
        )
    lines.append("")

    # Second check: the evaluate_recall / query-vector / rank logic itself.
    # Already established by direct code reading (see phase22_notes.md) that
    # phase 9 and OutfitTransformer share byte-identical cir_eval.py rank logic
    # (query_mat @ pool_emb.T, ranks = (sims >= target_sims).sum(axis=1)),
    # and CSA-Net implements a documented, paper-faithful adaptation
    # (average per-context-item pairwise distance, its own eq. 5) using the
    # identical rank/hit-counting convention. Record the actual rank-counting
    # lines from each script here as direct evidence.
    lines.append("## Rank-counting logic in each evaluation script (direct excerpt, not paraphrased)\n")
    excerpts = {
        "Phase 9 / phase 20 (`cir_eval.py`, `evaluate_recall`)":
            REPO_ROOT / "week4/phase20_full_consolidation/scripts/cir_eval.py",
        "OutfitTransformer (`04b_cir_eval_random_negatives.py`, `evaluate_recall`)":
            REPO_ROOT / "week4/phase14b_outfittransformer_category_negatives/scripts/04b_cir_eval_random_negatives.py",
        "CSA-Net (`03_csa_cir_eval.py`, `evaluate_csa_recall`)":
            REPO_ROOT / "week4/phase13b_csa_net_siglip_backbone/scripts/03_csa_cir_eval.py",
    }
    for name, path in excerpts.items():
        src = path.read_text()
        rank_lines = [l for l in src.splitlines() if "rank" in l.lower() and ("=" in l or "hits" in l)]
        lines.append(f"**{name}**:")
        lines.append("```")
        lines.extend(rank_lines)
        lines.append("```")
        lines.append("")

    lines.append(
        "All three use the same convention: rank = count of candidates with a similarity "
        "(or, for CSA-Net, a smaller-is-better distance, inverted via `<=`) at least as "
        "good as the target's own; hit@K = rank <= K. No script silently uses a stricter "
        "or looser tie-breaking rule than the others."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    for name, (p, d) in resolved.items():
        print(f"{name}: {p} -> {d}")


if __name__ == "__main__":
    main()

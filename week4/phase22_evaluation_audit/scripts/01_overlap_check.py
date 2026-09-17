"""
Phase 22, step 1: quantify train/test item overlap directly against the
actual artifacts each model's training and the CIR benchmark's construction
produced -- Polyvore's official "nondisjoint" split allows the same physical
item to appear in both train and test outfits (just in different outfit
groupings), so this is expected structurally, not a bug in itself. The
question this script answers is whether the overlap is roughly symmetric
across the three models being compared (phase 9, CSA-Net/phase13b,
OutfitTransformer/phase14b run 2), since all three ultimately draw their
training items from the same two source files
(week3/.../polyvore_raw/nondisjoint/{train,valid}.json), via two independent
re-derivations (phase 9's own 02_build_training_pairs.py, and phase 13's
01_prepare_training_data.py, reused unchanged by phase 13b/14/14b).
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12_DIR = BASE_DIR.parent / "phase12_controllable_modes"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"

CIR_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
PHASE9_EDGES = PHASE9_DIR / "data" / "positive_edges.json"
PHASE13_TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"

OUT_MD = BASE_DIR / "overlap_check.md"


def load_benchmark_items():
    with open(CIR_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    pool_items = set()
    for items in pools.values():
        pool_items.update(items)

    query_target_items = {q["target_item"] for q in queries}
    query_context_items = set()
    for q in queries:
        query_context_items.update(q["query_items"])

    all_benchmark_items = pool_items | query_target_items | query_context_items
    return {
        "pool_items": pool_items,
        "query_target_items": query_target_items,
        "query_context_items": query_context_items,
        "all_benchmark_items": all_benchmark_items,
    }


def load_phase9_train_items():
    with open(PHASE9_EDGES) as f:
        edges = json.load(f)
    train_items, val_items = set(), set()
    for e in edges:
        if e["split"] == "train":
            train_items.add(e["source"])
            train_items.add(e["target"])
        else:
            val_items.add(e["source"])
            val_items.add(e["target"])
    return train_items, val_items


def load_phase13_train_items():
    with open(PHASE13_TRAINING_DATA) as f:
        td = json.load(f)
    train_items = set()
    for items in td["train_items_by_category"].values():
        train_items.update(items)
    val_items = set()
    for items in td["val_items_by_category"].values():
        val_items.update(items)
    return train_items, val_items


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    bench = load_benchmark_items()
    all_bench = bench["all_benchmark_items"]
    pool_items = bench["pool_items"]
    target_items = bench["query_target_items"]

    p9_train, p9_val = load_phase9_train_items()
    p13_train, p13_val = load_phase13_train_items()

    # Sanity: phase9's own item extraction vs phase13's independent
    # re-derivation from the SAME source files should produce a near-identical
    # train+val item universe (both filter on "has downloaded image + known
    # semantic_category", confirmed by reading both scripts).
    p9_all = p9_train | p9_val
    p13_all = p13_train | p13_val
    universe_overlap = len(p9_all & p13_all)
    universe_union = len(p9_all | p13_all)
    universe_jaccard = universe_overlap / universe_union if universe_union else 0.0

    configs = {
        "Phase 9 (ProjectionHead)": (p9_train, p9_val),
        "Phase 13/13b (CSA-Net) & Phase 14/14b (OutfitTransformer) -- shared training_data.json": (p13_train, p13_val),
    }

    lines = [
        "# Phase 22, Step 1: Train/Test Item Overlap Against the CIR Benchmark",
        "",
        "Polyvore's official 'nondisjoint' split explicitly allows the same physical "
        "item to appear in both train and test outfits (as part of different outfit "
        "groupings) -- this is a known, documented property of the split, not a leakage "
        "bug to be fixed. What matters for a fair comparison is whether this overlap is "
        "roughly symmetric across the models being compared, since all three were "
        "trained on the identical official train/valid split files.",
        "",
        f"**CIR benchmark item universe** (from `{CIR_BENCHMARK.relative_to(BASE_DIR.parent.parent)}`): "
        f"{len(all_bench)} unique items appear in the benchmark (queries' target items, "
        f"queries' own context items, and/or category candidate pools). "
        f"{len(pool_items)} unique items sit in some candidate pool; "
        f"{len(target_items)} unique items are ever a query's target.",
        "",
        "## Sanity check: do phase 9's own item extraction and phase 13's independent "
        "re-derivation agree on the training item universe?",
        "",
        f"Both pipelines start from the same two files "
        f"(`polyvore_raw/nondisjoint/train.json` + `valid.json`) and apply the same "
        f"qualifying filter (downloaded image + known `semantic_category`), but via two "
        f"separately written scripts (`week3/.../02_build_training_pairs.py` vs "
        f"`week4/phase13_csa_net_baseline/scripts/01_prepare_training_data.py`).",
        "",
        f"- Phase 9's train+val item universe: {len(p9_all)} unique items",
        f"- Phase 13's train+val item universe: {len(p13_all)} unique items",
        f"- Overlap: {universe_overlap} items shared, {universe_union} in the union "
        f"-- Jaccard = {universe_jaccard:.4f}",
        "",
    ]
    if universe_jaccard > 0.98:
        lines.append(
            "**Confirmed: the two independent extractions agree almost exactly.** "
            "This rules out a subtle difference in how the two pipelines read the "
            "official split (e.g. a different image-validity filter) as a source of "
            "asymmetric overlap below -- both models are drawing from essentially the "
            "same training item pool."
        )
    else:
        lines.append(
            f"**The two extractions diverge more than expected** (Jaccard={universe_jaccard:.4f}, "
            "not near 1.0) -- this needs to be understood before trusting the overlap "
            "percentages below as comparable across models, since it means phase 9 and "
            "the CSA-Net/OutfitTransformer reproductions were not necessarily trained on "
            "the same item pool even though they read the same source files."
        )
    lines.append("")

    lines.append("## Overlap between each model's training items and the CIR benchmark's item universe\n")
    lines.append("| Model | Train items seen | Val items seen | Train ∩ benchmark | % of benchmark items | Val ∩ benchmark | % of benchmark items |")
    lines.append("|---|---|---|---|---|---|---|")
    results = {}
    for name, (train_items, val_items) in configs.items():
        train_overlap = train_items & all_bench
        val_overlap = val_items & all_bench
        results[name] = {
            "n_train": len(train_items), "n_val": len(val_items),
            "train_overlap": len(train_overlap), "train_overlap_pct": pct(len(train_overlap), len(all_bench)),
            "val_overlap": len(val_overlap), "val_overlap_pct": pct(len(val_overlap), len(all_bench)),
        }
        lines.append(
            f"| {name} | {len(train_items)} | {len(val_items)} | {len(train_overlap)} "
            f"| {pct(len(train_overlap), len(all_bench)):.1f}% | {len(val_overlap)} "
            f"| {pct(len(val_overlap), len(all_bench)):.1f}% |"
        )
    lines.append("")

    # Direct symmetry check: phase 9 vs the shared phase13 training_data.json
    p9_train_overlap = p9_train & all_bench
    p13_train_overlap = p13_train & all_bench
    both = p9_train_overlap & p13_train_overlap
    only_p9 = p9_train_overlap - p13_train_overlap
    only_p13 = p13_train_overlap - p9_train_overlap

    lines.append("## Direct symmetry check: is the overlap the same set of items for both training pipelines?\n")
    lines.append(
        f"- Benchmark items overlapping phase 9's train set: {len(p9_train_overlap)}\n"
        f"- Benchmark items overlapping phase 13's (CSA-Net/OutfitTransformer) train set: {len(p13_train_overlap)}\n"
        f"- In both: {len(both)}\n"
        f"- Only in phase 9's overlap: {len(only_p9)}\n"
        f"- Only in phase 13's overlap: {len(only_p13)}\n"
    )
    symmetric = len(only_p9) + len(only_p13) < 0.05 * max(len(p9_train_overlap), len(p13_train_overlap), 1)
    if symmetric:
        lines.append(
            "**The overlapping item sets are nearly identical between the two training "
            "pipelines.** Since phase 9, CSA-Net (phase 13b), and OutfitTransformer "
            "(phase 14/14b) all draw from the same official train/valid split, and the "
            "resulting benchmark-overlap sets match this closely, no model has a "
            "structural advantage over the others purely from which items leaked across "
            "the split -- the leakage, where it exists, is shared."
        )
    else:
        lines.append(
            "**The overlapping item sets differ more than a small edge-case mismatch "
            "would explain.** This is flagged as a real asymmetry worth investigating "
            "further, not rounded away."
        )
    lines.append("")

    lines.append("## Training-procedure differences beyond raw item overlap\n")
    lines.append(
        "Even with a symmetric item-level overlap, one model could still benefit more "
        "from the shared overlap if it trains for materially more epochs/passes over "
        "the data, or samples the overlapping items more aggressively than the others. "
        "Checked directly against each phase's own training logs/notes:\n"
    )
    lines.append(
        "- **Phase 9**: `05_train_projection.py`, MAX_EPOCHS=100 with early stopping "
        "(PATIENCE=5, MIN_DELTA=1e-4) on val loss -- converged well short of 100 in "
        "every run this project has logged (see phase 9's own training curves).\n"
        "- **Phase 13b (CSA-Net)**: `01_train_full.py`, max_epochs=40 with early "
        "stopping (patience=5) on the *same* held-out val_outfits split as phase 9's "
        "val split source (both derive from `valid.json`).\n"
        "- **Phase 14b run 2 (OutfitTransformer)**: `02b_train_full_random_negatives.py`, "
        "same `training_data.json` train/val split, same early-stopping discipline.\n"
    )
    lines.append(
        "All three use early stopping keyed to validation loss on the same underlying "
        "val_outfits pool (from `valid.json`), rather than a fixed epoch budget that "
        "could let one model see more repetitions of overlapping items than another. No "
        "model in this comparison was trained with an unusually large epoch count, extra "
        "data augmentation, or an oversampling scheme that would let it exploit shared "
        "train/test overlap more than the others."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    print(f"Benchmark universe: {len(all_bench)} items")
    for name, r in results.items():
        print(f"{name}: train_overlap={r['train_overlap']} ({r['train_overlap_pct']:.1f}%)")


if __name__ == "__main__":
    main()

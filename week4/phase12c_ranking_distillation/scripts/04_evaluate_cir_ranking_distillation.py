"""
Phase 12c, step 4.1: re-run the exact same CIR benchmark evaluation as phase
12/12b (identical harness, pointed directly at phase 12's cir_benchmark.json)
on the ranking-distillation checkpoint, with phase 12 and phase 12b's own
numbers carried forward for a three-way before/after/after comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark
from model import ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT = BASE_DIR / "models" / "ranking_distillation.pt"
RESULTS_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KS = (10, 30, 50)

# From week4/phase12_controllable_modes/results_table.md and
# week4/phase12b_controllable_modes_fixed/results_table.md (those phases' own runs).
PHASE12_NUMBERS = {
    "Raw SigLIP (alone)": {10: 0.0553, 30: 0.1067, 50: 0.1437},
    "Phase 9 Model A (alone)": {10: 0.1317, 30: 0.2464, 50: 0.3216},
    "Substitute mode": {10: 0.1329, 30: 0.2467, 50: 0.3215},
    "Complement mode": {10: 0.1335, 30: 0.2507, 50: 0.3250},
    "Blend (0.5)": {10: 0.1366, 30: 0.2524, 50: 0.3289},
}
PHASE12B_NUMBERS = {
    "Substitute mode": {10: 0.1212, 30: 0.2254, 50: 0.2957},
    "Complement mode": {10: 0.1200, 30: 0.2262, 50: 0.2963},
    "Blend (0.5)": {10: 0.1207, 30: 0.2261, 50: 0.2957},
}
LITERATURE_ANCHOR = {10: 0.0958, 30: 0.1796, 50: 0.2198}


def load_raw_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


@torch.no_grad()
def project(model, embeddings, alpha, batch_size=4096):
    outs = []
    for start in range(0, len(embeddings), batch_size):
        x = torch.tensor(embeddings[start:start + batch_size], device=DEVICE)
        outs.append(model(x, alpha=alpha).cpu().numpy())
    return np.concatenate(outs, axis=0)


def fmt_row(label, recall, n=None):
    n_str = n if n is not None else "--"
    return f"| {label} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} | {n_str} |"


def main():
    pools, queries = load_benchmark()
    item_ids, raw_emb = load_raw_embeddings()

    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    configs = [("Substitute mode", 1.0), ("Complement mode", 0.0), ("Blend (0.5)", 0.5)]
    new_rows = {}
    for label, alpha in configs:
        emb = project(model, raw_emb, alpha)
        recall, n, n_skip = evaluate_recall(pools, queries, item_ids, emb, ks=KS)
        new_rows[label] = (recall, n, n_skip)
        print(f"{label}: {recall} (n={n}, {n_skip} skipped)")

    with open(BASE_DIR / "data" / "phase12c_recall.json", "w") as f:
        json.dump({label: {"recall": {str(k): v for k, v in r.items()}, "n_queries": n, "n_skipped": s}
                   for label, (r, n, s) in new_rows.items()}, f, indent=2)

    lines = [
        "# Phase 12c: CIR Benchmark Results, Three-Way Comparison",
        "",
        f"Same benchmark as phase 12/12b (`week4/phase12_controllable_modes/data/cir_benchmark.json`, "
        f"unchanged): {sum(len(v) for v in pools.values())} pool slots, {len(queries)} queries.",
        "",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 | N queries |",
        "|---|---|---|---|---|",
    ]
    lines.append(fmt_row("Raw SigLIP (alone) [reference]", PHASE12_NUMBERS["Raw SigLIP (alone)"], 29681))
    lines.append(fmt_row("Phase 9 Model A (alone) [reference]", PHASE12_NUMBERS["Phase 9 Model A (alone)"], 29681))
    lines.append("| | | | | |")
    for label, alpha in configs:
        lines.append(fmt_row(f"Phase 12 (batch-local pairwise): {label}", PHASE12_NUMBERS[label], 29681))
        lines.append(fmt_row(f"Phase 12b (PCA-128 target): {label}", PHASE12B_NUMBERS[label], 29681))
        new_recall, n, n_skip = new_rows[label]
        lines.append(fmt_row(f"**Phase 12c (ranking distillation): {label}**", new_recall, n))
        lines.append("| | | | | |")
    lines.append(fmt_row("OutfitTransformer (literature anchor, not independently reproduced)", LITERATURE_ANCHOR))
    lines.append("")
    lines.append(
        "**Reading this table**: as established in phases 12 and 12b, Recall@K alone is not "
        "sensitive enough to tell whether the two modes are behaviorally distinct -- both "
        "prior phases showed substitute and complement scoring almost identically here even "
        "when their actual retrieved item sets differed substantially. See "
        "`control_effectiveness.md` for the decisive checks (top-10 overlap between modes, "
        "and critically, each mode's overlap with RAW SigLIP's own retrieval -- the check "
        "that actually settles whether this fix worked)."
    )
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")


if __name__ == "__main__":
    main()

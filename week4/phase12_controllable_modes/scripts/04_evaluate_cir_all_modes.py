"""
Phase 12, step 4.1: evaluate substitute mode alone, complement mode alone,
and an interpolated blend (alpha=0.5) on the CIR benchmark, alongside the
step-2 baselines (raw SigLIP, phase 9 Model A) and the OutfitTransformer
literature anchor -- all in one results table.
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
CHECKPOINT = BASE_DIR / "models" / "controllable_modes.pt"
BASELINE_JSON = BASE_DIR / "data" / "baseline_recall.json"
RESULTS_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KS = (10, 30, 50)


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


def fmt_row(label, recall, n, n_skip):
    return f"| {label} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} | {n if n is not None else '--'} |"


def main():
    pools, queries = load_benchmark()
    item_ids, raw_emb = load_raw_embeddings()

    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    configs = [
        ("Substitute mode alone (alpha=1.0)", 1.0),
        ("Complement mode alone (alpha=0.0)", 0.0),
        ("Interpolated blend (alpha=0.5)", 0.5),
    ]

    rows = {}
    for label, alpha in configs:
        emb = project(model, raw_emb, alpha)
        recall, n, n_skip = evaluate_recall(pools, queries, item_ids, emb, ks=KS)
        rows[label] = (recall, n, n_skip)
        print(f"{label}: {recall} (n={n}, {n_skip} skipped)")

    lines = [
        "# Phase 12: CIR Benchmark Results",
        "",
        f"Benchmark: `data/cir_benchmark.json`, {sum(len(v) for v in pools.values())} pool "
        f"slots across 11 categories, {len(queries)} leave-one-out queries kept "
        "(see `cir_protocol_notes.md` for exactly how the pools/queries were built).",
        "",
        "## Recall@K, all configurations",
        "",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 | N queries |",
        "|---|---|---|---|---|",
    ]
    b = baseline["Raw SigLIP (alone)"]
    lines.append(fmt_row("Raw SigLIP (alone)", {int(k): v for k, v in b["recall"].items()}, b["n_queries"], b["n_skipped"]))
    b = baseline["Phase 9 Model A (Polyvore-trained)"]
    lines.append(fmt_row("Phase 9 Model A (Polyvore-trained, alone)", {int(k): v for k, v in b["recall"].items()}, b["n_queries"], b["n_skipped"]))
    for label, alpha in configs:
        recall, n, n_skip = rows[label]
        lines.append(fmt_row(f"Phase 12: {label}", recall, n, n_skip))
    anchor_key = "OutfitTransformer (literature anchor, not independently reproduced -- see cir_protocol_notes.md)"
    b = baseline[anchor_key]
    lines.append(fmt_row(anchor_key, {int(k): v for k, v in b["recall"].items()}, None, None))
    lines.append("")
    lines.append(
        "**Caveat on the literature-anchor row**: this project's benchmark (leave-one-out "
        "over every qualifying test item-slot, per-category pools capped at 3,000 with "
        "queries whose target missed the cap dropped -- see `cir_protocol_notes.md`) is "
        "built independently of OutfitTransformer's own exact candidate-pool construction, "
        "since no usable reference implementation of their retrieval eval was found. "
        "Comparing *rankings* across configurations measured on this project's own harness "
        "(the rows above raw SigLIP/Model A/substitute/complement/blend) is a clean, "
        "apples-to-apples comparison; comparing *absolute magnitude* against the literature "
        "row is not, for the same reason phase 4's DeepFashion-vs-Amazon comparison flagged "
        "absolute-magnitude comparisons across different retrieval setups as unsafe."
    )
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")


if __name__ == "__main__":
    main()

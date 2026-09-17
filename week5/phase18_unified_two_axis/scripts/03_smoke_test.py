"""Phase 18 smoke test: small subset, few epochs, confirm no collapse in
ANY of the 4 heads before committing to a full run. Same collapse-guard
threshold precedent as phases 16/16c/16d (mean pairwise cosine > 0.98 on a
sample = warning sign)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FourHeadDedicatedCapacity, mean_pairwise_cosine
from train_core import (
    BATCH_SIZE, CORNERS, DEVICE, LR, SEED, WEIGHT_DECAY, build_negatives_for_edges, load_data,
    run_epoch_train,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = BASE_DIR / "logs" / "smoke_test_report.md"

N_EPOCHS = 3
SUBSET_REL = 5000
SUBSET_TAIL = 3000


def main():
    (embeddings, idx, item_ids, rel_train, rel_val, tail_train, tail_val, all_positive_targets,
     sr_nn_indices, sr_nn_sims, st_nn_indices, st_nn_sims) = load_data()

    rng = random.Random(SEED)
    rel_subset = rng.sample(rel_train, min(SUBSET_REL, len(rel_train)))
    tail_subset = rng.sample(tail_train, min(SUBSET_TAIL, len(tail_train)))
    print(f"Smoke test: {len(rel_subset)} relevance-axis edges, {len(tail_subset)} tail-axis edges, "
          f"{N_EPOCHS} epochs, device={DEVICE}")

    torch.manual_seed(SEED)
    model = FourHeadDedicatedCapacity().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = {c: 1.0 for c in CORNERS}  # uncalibrated on purpose -- smoke test just checks for collapse

    for epoch in range(N_EPOCHS):
        train_losses = run_epoch_train(model, optimizer, embeddings, idx, rel_subset, tail_subset,
                                        all_positive_targets, item_ids, sr_nn_indices, sr_nn_sims,
                                        st_nn_indices, st_nn_sims, rng, weights, log_every=0)
        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=256, replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse = {c: mean_pairwise_cosine(model.single_head_output(sample_emb, c)) for c in CORNERS}
        summary = " ".join(f"{c}={train_losses[c]:.4f}" for c in CORNERS)
        collapse_summary = " ".join(f"{c}={collapse[c]:.4f}" for c in CORNERS)
        print(f"epoch {epoch}: losses[{summary}] collapse[{collapse_summary}]")

    any_collapse = any(v > 0.98 for v in collapse.values())
    lines = [
        "# Phase 18: Smoke Test Report",
        "",
        f"{len(rel_subset)} relevance-axis / {len(tail_subset)} tail-axis edges, {N_EPOCHS} epochs, "
        "uncalibrated equal weights (1.0 each) -- this test only checks for embedding collapse in any "
        "of the 4 heads before committing to full training with fresh calibration.",
        "",
        "## Final-epoch mean pairwise cosine per head (256-item sample)",
        "",
        "| Corner | Mean pairwise cosine |",
        "|---|---|",
    ] + [f"| {c} | {collapse[c]:.4f} |" for c in CORNERS] + [
        "",
        f"**Collapse warning (>0.98 in any head): {'YES -- investigate before full training' if any_collapse else 'NO'}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")
    print(f"Collapse warning: {any_collapse}")


if __name__ == "__main__":
    main()

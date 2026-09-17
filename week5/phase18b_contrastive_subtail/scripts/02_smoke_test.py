"""Phase 18b smoke test: small subset, few epochs, confirm no collapse in
ANY of the 4 heads (especially the newly-reformulated sub_tail) before
committing to a full run."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FourHeadDedicatedCapacity, mean_pairwise_cosine
from train_core import CORNERS, DEVICE, LR, SEED, WEIGHT_DECAY, load_data, run_epoch_train

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = BASE_DIR / "logs" / "smoke_test_report.md"

N_EPOCHS = 3
SUBSET_REL = 5000
SUBSET_CT = 3000
SUBSET_ST = 5000


def main():
    (embeddings, idx, item_ids, rel_train, rel_val, ct_train, ct_val, st_train, st_val, all_positive_targets,
     sr_nn_indices, sr_nn_sims, head_tier_ids) = load_data()

    rng = random.Random(SEED)
    rel_subset = rng.sample(rel_train, min(SUBSET_REL, len(rel_train)))
    ct_subset = rng.sample(ct_train, min(SUBSET_CT, len(ct_train)))
    st_subset = rng.sample(st_train, min(SUBSET_ST, len(st_train)))
    print(f"Smoke test: {len(rel_subset)} REL / {len(ct_subset)} CT / {len(st_subset)} ST edges, "
          f"{N_EPOCHS} epochs, device={DEVICE}")

    torch.manual_seed(SEED)
    model = FourHeadDedicatedCapacity().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = {c: 1.0 for c in CORNERS}  # uncalibrated on purpose -- smoke test just checks for collapse

    for epoch in range(N_EPOCHS):
        train_losses = run_epoch_train(model, optimizer, embeddings, idx, rel_subset, ct_subset, st_subset,
                                        all_positive_targets, item_ids, head_tier_ids, sr_nn_indices, sr_nn_sims,
                                        rng, weights, log_every=0)
        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=256, replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse = {c: mean_pairwise_cosine(model.single_head_output(sample_emb, c)) for c in CORNERS}
        summary = " ".join(f"{c}={train_losses[c]:.4f}" for c in CORNERS)
        collapse_summary = " ".join(f"{c}={collapse[c]:.4f}" for c in CORNERS)
        print(f"epoch {epoch}: losses[{summary}] collapse[{collapse_summary}]")

    any_collapse = any(v > 0.98 for v in collapse.values())
    lines = [
        "# Phase 18b: Smoke Test Report",
        "",
        f"{len(rel_subset)} REL / {len(ct_subset)} CT / {len(st_subset)} ST edges, {N_EPOCHS} epochs, "
        "uncalibrated equal weights (1.0 each) -- checks for collapse in any of the 4 heads, especially "
        "the newly-reformulated sub_tail (now MNRL with head-biased negatives instead of ranking "
        "distillation), before committing to full training with fresh calibration.",
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

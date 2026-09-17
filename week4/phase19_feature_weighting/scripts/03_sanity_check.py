"""
Phase 19, step 4: confirm the learned weight vector didn't collapse to
near-zero (the mechanism becomes a no-op / degenerate scaling) or become
uniform across all dimensions (if every dimension gets the same weight,
then z = normalize(x * c) = normalize(x) for any positive scalar c --
L2-normalization erases a uniform rescale entirely, so a uniform weight
vector means the mechanism learned literally nothing beyond raw SigLIP,
even though the loss might still have gone down from other effects). A
lightweight check given how few parameters this mechanism has (768, vs.
the tens of thousands in every other mechanism this project has trained),
but still confirmed directly, not assumed.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FeatureWeighting

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PT = BASE_DIR / "models" / "feature_weighting.pt"
OUT_MD = BASE_DIR / "sanity_check.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# thresholds: near-zero collapse would put the mean well below 1 (or the
# whole vector near 0); uniformity would put std near 0 (all 768 dims
# converging to the identical value).
NEAR_ZERO_MEAN_THRESHOLD = 0.05
UNIFORM_STD_THRESHOLD = 1e-4


def main():
    model = FeatureWeighting().to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    w = model.weight.detach().cpu().numpy()

    mean, std, wmin, wmax = float(w.mean()), float(w.std()), float(w.min()), float(w.max())
    frac_near_zero = float((np.abs(w) < 0.05).mean())

    collapsed_to_zero = mean < NEAR_ZERO_MEAN_THRESHOLD and wmax < NEAR_ZERO_MEAN_THRESHOLD * 5
    uniform = std < UNIFORM_STD_THRESHOLD

    lines = [
        "# Phase 19, Step 4: Weight Vector Sanity Check",
        "",
        f"Learned weight vector (768 dims), from the best (early-stopped) checkpoint:",
        "",
        f"- mean = {mean:.4f}",
        f"- std = {std:.4f}",
        f"- min = {wmin:.4f}",
        f"- max = {wmax:.4f}",
        f"- fraction of dims with |weight| < 0.05 (near-zero, effectively dropped): {frac_near_zero:.2%}",
        "",
        f"Init value was exactly 1.0 for every dimension (ones-init, an exact identity map before training).",
        "",
        f"**Near-zero collapse check**: {'FAILED -- weight vector collapsed toward zero' if collapsed_to_zero else 'PASSED -- mean and max are well away from zero'}.",
        f"**Uniformity check**: {'FAILED -- std is near zero, all dimensions converged to the same value (a no-op under L2-normalization)' if uniform else f'PASSED -- std ({std:.4f}) is a real, non-trivial spread across dimensions, not a uniform rescale'}.",
        "",
    ]
    if not collapsed_to_zero and not uniform:
        lines.append("**Overall: no sign of collapse or degeneracy. The mechanism learned a real, "
                      "non-trivial per-dimension reweighting, not a no-op.**")
    else:
        lines.append("**Overall: a real problem was found -- see the failed check(s) above. "
                      "Reported directly per the brief's instruction, not proceeding to treat "
                      "the eval result as informative about a genuinely-learned mechanism.**")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()

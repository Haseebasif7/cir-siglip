"""
Phase 12d, step 2: check monotonicity of each of the four diagnostic metrics
across the full 11-point alpha sweep, not just at the two endpoints. A metric
that's correct at alpha=0 and alpha=1 but reverses direction somewhere in the
middle would mean the mechanism isn't actually a usable dial, even though the
endpoints work -- that's exactly the failure mode this check is built to
catch, checked directly rather than eyeballed from the table.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE_DIR / "data" / "alpha_sweep_results.json"
OUT_MD = BASE_DIR / "monotonicity_check.md"

ALPHAS = [round(0.1 * i, 1) for i in range(11)]

# (metric key, expected direction as alpha increases, human label)
METRICS = [
    ("axis1_visual_sim", "increase", "Visual similarity to query (a)"),
    ("axis2_hit_rate", "decrease", "Co-occurrence hit rate (b)"),
    ("overlap_with_raw", "increase", "Overlap with raw SigLIP (c)"),
    ("overlap_with_complement", "decrease", "Overlap with complement mode (d)"),
]
# A step-to-step change smaller than this (in the metric's own units) is
# treated as flat/noise, not a violation, since these are sample-based
# estimates (n=500-1000) with inherent sampling noise -- flagging every
# sub-noise-floor wiggle as "non-monotonic" would overstate real problems.
# Set from this project's own sample sizes: for a proportion metric on
# n=1000, the binomial standard error at p~0.1-0.5 is roughly 0.01-0.016;
# 0.01 is used as a conservative (lower, stricter) noise floor.
NOISE_FLOOR = 0.01


def check_metric(values, expected_direction):
    diffs = np.diff(values)
    if expected_direction == "increase":
        violations = [(i, d) for i, d in enumerate(diffs) if d < -NOISE_FLOOR]
    else:
        violations = [(i, d) for i, d in enumerate(diffs) if d > NOISE_FLOOR]
    net_change = values[-1] - values[0]
    endpoint_correct = (net_change > 0) if expected_direction == "increase" else (net_change < 0)
    return violations, net_change, endpoint_correct


def check_extremum_location(values, alphas, expected_direction):
    """A per-step noise floor can miss a real hump/dip made of several small
    steps in a row that never individually exceed the floor -- e.g. a metric
    that's expected to decrease monotonically but actually rises gently for
    several steps before falling. A robust, step-noise-independent check: for
    a truly monotonic metric, the GLOBAL min (if increasing) or max (if
    decreasing) must sit at alpha=0.0. If it sits in the interior instead,
    that is real evidence of non-monotonicity regardless of how small any
    single step's move was."""
    if expected_direction == "increase":
        extremum_idx = int(np.argmin(values))
        extremum_kind = "minimum"
    else:
        extremum_idx = int(np.argmax(values))
        extremum_kind = "maximum"
    at_start = extremum_idx == 0
    return at_start, extremum_kind, alphas[extremum_idx], values[extremum_idx]


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    lines = [
        "# Phase 12d, Step 2: Monotonicity Check",
        "",
        f"For each metric, checked step-by-step across all 11 alpha values (0.0 to 1.0, "
        f"step 0.1) whether it moves in the expected direction consistently, not just "
        f"between the two endpoints. A step-to-step change smaller than {NOISE_FLOOR} is "
        "treated as noise/flat, not a violation, given the underlying 500-1,000-query "
        "sample sizes (see script docstring for the noise-floor justification).",
        "",
    ]

    any_step_violation = False
    any_hump = False
    for key, direction, label in METRICS:
        values = [results[str(a)][key] for a in ALPHAS]
        violations, net_change, endpoint_correct = check_metric(values, direction)
        at_start, extremum_kind, extremum_alpha, extremum_val = check_extremum_location(values, ALPHAS, direction)

        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"Values across alpha 0.0 -> 1.0: {[round(v, 4) for v in values]}")
        lines.append(f"Expected direction as alpha rises: **{direction}**. "
                      f"Net change (alpha=1.0 minus alpha=0.0): {net_change:+.4f} "
                      f"({'correct' if endpoint_correct else 'WRONG'} endpoint direction).")
        if violations:
            any_step_violation = True
            viol_str = ", ".join(f"alpha {ALPHAS[i]:.1f}->{ALPHAS[i+1]:.1f} (delta {d:+.4f})" for i, d in violations)
            lines.append(f"**{len(violations)} single step(s) move against the expected direction "
                         f"beyond the {NOISE_FLOOR} noise floor**: {viol_str}")
        else:
            lines.append(f"No single step moves against the expected direction beyond the "
                         f"{NOISE_FLOOR} noise floor.")
        if at_start:
            lines.append(f"Global {extremum_kind} sits at alpha=0.0 (the expected endpoint) -- "
                         "no multi-step hump/dip either.")
        else:
            any_hump = True
            lines.append(f"**Global {extremum_kind} sits at alpha={extremum_alpha:.1f} "
                         f"({extremum_val:.4f}), NOT at the expected alpha=0.0 endpoint** -- "
                         "even though no single step exceeded the noise floor, the metric "
                         "drifts the wrong way across several consecutive small steps before "
                         "turning around, a real (if gentle) hump/dip a pure step-wise check "
                         "would miss.")
        lines.append("")

    lines.append("## Overall verdict")
    lines.append("")
    if any_step_violation or any_hump:
        parts = []
        if any_step_violation:
            parts.append("at least one metric has a single step that reverses beyond the noise floor")
        if any_hump:
            parts.append("at least one metric has a real multi-step hump/dip even where no single "
                          "step alone exceeded the noise floor")
        lines.append(f"**Not cleanly monotonic across the full range** -- {'; and '.join(parts)}. "
                      "See `../phase12d_notes.md` for what this means for whether the mechanism "
                      "is a genuine, usable dial across its full range.")
    else:
        lines.append("**All four metrics move consistently in their expected direction across "
                      "the full sweep, not just between the endpoints, and no metric's global "
                      "extremum sits in the interior** -- real evidence this is a genuine dial, "
                      "not just two points that happen to work. See `../phase12d_notes.md` for "
                      "the full verdict alongside the step 3 retrieval-level smoothness check.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()

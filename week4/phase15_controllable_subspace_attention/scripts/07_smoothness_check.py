"""
Phase 15, step 6: adjacent-vs-distant top-K overlap smoothness check, same
methodology as phase 12d's `03_adjacent_vs_distant_overlap.py` -- builds the
full 11x11 pairwise top-K overlap matrix across alpha values from a
checkpoint's `topk_cache_<label>.json`, buckets by |delta alpha|, and checks
whether adjacent alphas retrieve more similar sets than distant ones (a
"genuine usable dial") using the same thresholds phase 12d declared:
gap > 0.15 and monotonic decay -> "real, smooth relationship confirmed";
gap > 0.05 -> "partial confirmation"; else -> "not confirmed".
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
TOP_K = 10


def overlap(a, b, k=TOP_K):
    return len(set(a) & set(b)) / k


def run(label):
    cache_path = BASE_DIR / "data" / f"topk_cache_{label}.json"
    with open(cache_path) as f:
        cache = json.load(f)

    n = len(ALPHAS)
    matrix = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        ai, aj = str(ALPHAS[i]), str(ALPHAS[j])
        common_qis = set(cache[ai].keys()) & set(cache[aj].keys())
        vals = [overlap(cache[ai][qi], cache[aj][qi]) for qi in common_qis]
        if vals:
            matrix[i, j] = matrix[j, i] = float(np.mean(vals))

    by_delta = {}
    for i, j in combinations(range(n), 2):
        delta = round(abs(ALPHAS[i] - ALPHAS[j]), 1)
        if not np.isnan(matrix[i, j]):
            by_delta.setdefault(delta, []).append(matrix[i, j])
    delta_means = {d: float(np.mean(vs)) for d, vs in sorted(by_delta.items())}

    deltas_sorted = sorted(delta_means.keys())
    decay_vals = [delta_means[d] for d in deltas_sorted]
    monotonic_decay = all(decay_vals[k] >= decay_vals[k + 1] - 0.02 for k in range(len(decay_vals) - 1))

    adjacent_mean = delta_means.get(0.1, float("nan"))
    distant_mean = delta_means.get(1.0, float("nan"))
    gap = adjacent_mean - distant_mean

    if gap > 0.15 and monotonic_decay:
        verdict = "Real, smooth relationship confirmed"
    elif gap > 0.05:
        verdict = "Partial confirmation"
    else:
        verdict = "Not confirmed"

    print(f"[{label}] delta_means: {delta_means}")
    print(f"[{label}] adjacent_mean(0.1)={adjacent_mean:.4f} distant_mean(1.0)={distant_mean:.4f} "
          f"gap={gap:.4f} monotonic_decay={monotonic_decay} -> {verdict}")

    out = {
        "label": label, "delta_means": delta_means, "adjacent_mean": adjacent_mean,
        "distant_mean": distant_mean, "gap": gap, "monotonic_decay": monotonic_decay,
        "verdict": verdict,
    }
    out_path = BASE_DIR / "data" / f"smoothness_check_{label}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{label}] Saved {out_path}")
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "continuous"
    run(label)

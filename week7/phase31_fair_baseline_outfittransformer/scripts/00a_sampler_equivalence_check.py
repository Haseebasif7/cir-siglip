"""
Phase 31, step 0: verify the vectorized negative sampler (index-space
rejection sampling on precomputed per-category global-index arrays) produces
the SAME distribution as phase 14b's original per-sample Python list
comprehension (`TrainState._sample_negatives`, negative_mode="random") --
a speed-only refactor, not a change to the negative-sampling scheme itself.

Original (train_core.py, random mode): draws num_negatives items uniformly
without replacement from `pool[category]`, excluding the outfit's own items,
via `extra_pool = [i for i in pool[category] if i not in exclude]` then
`rng.sample(extra_pool, k)` -- O(pool size) Python-level work per sample
(mean pool size 36,513), the measured ~56s/epoch bottleneck.

Vectorized (this phase): precompute `cat_pool_gidx[cat]` once as a numpy
int64 array; per sample, draw candidate positions via `rng.integers` in
small batches and reject/redraw only against the tiny (~3-8 item) exclude
set -- O(k) Python-level work per sample, independent of pool size.

Check: for one fixed (target, category, exclude) triple, draw N=1,000,000
negatives under each implementation and confirm (a) zero excluded items ever
appear under either, (b) the empirical frequency distribution over the pool
is statistically indistinguishable (chi-square test, not significant).
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA = REPO_ROOT / "week4/phase13_csa_net_baseline/data/training_data.json"

N_DRAWS_SAMPLES = 20000   # number of independent (exclude-set) draws to average speed over
N_FREQ_DRAWS = 200000     # draws for the frequency/chi-square check (single fixed exclude set)
NUM_NEGATIVES = 10


def original_sample(rng, pool_items, exclude, num_negatives):
    """Exact reproduction of train_core.py's random-mode logic."""
    extra_pool = [i for i in pool_items if i not in exclude]
    return rng.sample(extra_pool, min(num_negatives, len(extra_pool)))


def vectorized_sample(rng_np, pool_gidx, exclude_gidx, num_negatives):
    n_pool = len(pool_gidx)
    k = min(num_negatives, max(n_pool - len(exclude_gidx), 0))
    if k <= 0:
        return np.array([], dtype=np.int64)
    chosen, seen = [], set()
    attempts = 0
    draw_size = min(n_pool, k + 8)
    while len(chosen) < k and attempts < 20:
        cand_pos = rng_np.integers(0, n_pool, size=draw_size)
        cand_gidx = pool_gidx[cand_pos]
        for g in cand_gidx:
            gi = int(g)
            if gi not in exclude_gidx and gi not in seen:
                seen.add(gi)
                chosen.append(gi)
                if len(chosen) >= k:
                    break
        attempts += 1
    return np.array(chosen, dtype=np.int64)


def main():
    print("Loading training_data.json...")
    with open(TRAINING_DATA) as f:
        td = json.load(f)
    items_by_cat = td["train_items_by_category"]
    cat = max(items_by_cat, key=lambda c: len(items_by_cat[c]))  # largest category, worst case for the original
    pool_items = items_by_cat[cat]  # a MULTISET -- an item appears once per outfit it's in (confirmed:
    # 51,132 entries, 36,481 unique for "shoes"). The original algorithm samples from this multiset, so
    # items appearing in more outfits are proportionally more likely to be drawn -- that's the real existing
    # distribution and must be preserved exactly, not flattened into uniform-over-unique-items.
    n_unique = len(set(pool_items))
    print(f"Using category '{cat}', pool size {len(pool_items)} ({n_unique} unique -- a real multiset)")

    # Canonical item -> global embedding index, built the way TrainState.idx really is (from the full,
    # deduplicated item universe in siglip_base.npz), NOT from this possibly-duplicated category list --
    # this is the fix for the bug the first run of this script caught (a naive `enumerate(pool_items)` dict
    # silently drops duplicate positions, last-write-wins, so an earlier duplicate's position wouldn't
    # resolve to the same canonical id as its later duplicate and could evade the exclude-set check).
    canonical_items = sorted(set(pool_items))
    item_to_gidx = {item: i for i, item in enumerate(canonical_items)}
    pool_gidx = np.array([item_to_gidx[item] for item in pool_items], dtype=np.int64)  # multiset preserved

    # A fixed, realistic exclude set: a handful of items from the SAME pool
    # (worst case for collision-checking -- in practice most excluded items
    # are the outfit's other members, usually different categories, but
    # testing same-category exclusion stresses the reject/redraw path harder).
    py_rng = random.Random(123)
    exclude_items = py_rng.sample(pool_items, 6)
    exclude_gidx = {item_to_gidx[i] for i in exclude_items}
    print(f"Exclude set: {len(exclude_items)} items")

    # --- Correctness: zero excluded items ever appear ---
    rng_orig = random.Random(42)
    rng_np = np.random.default_rng(42)
    orig_counter, vec_counter = Counter(), Counter()
    orig_violations, vec_violations = 0, 0

    t0 = time.time()
    for _ in range(N_FREQ_DRAWS):
        drawn = original_sample(rng_orig, pool_items, set(exclude_items), NUM_NEGATIVES)
        for d in drawn:
            if d in exclude_items:
                orig_violations += 1
            orig_counter[d] += 1
    t_orig = time.time() - t0

    t0 = time.time()
    for _ in range(N_FREQ_DRAWS):
        drawn = vectorized_sample(rng_np, pool_gidx, exclude_gidx, NUM_NEGATIVES)
        for d in drawn:
            item = canonical_items[int(d)]  # d is a canonical global index, not a pool-list position
            if item in exclude_items:
                vec_violations += 1
            vec_counter[item] += 1
    t_vec = time.time() - t0

    print(f"\nOriginal:   {N_FREQ_DRAWS} draws in {t_orig:.2f}s ({t_orig/N_FREQ_DRAWS*1000:.4f} ms/draw), "
          f"violations={orig_violations}")
    print(f"Vectorized: {N_FREQ_DRAWS} draws in {t_vec:.2f}s ({t_vec/N_FREQ_DRAWS*1000:.4f} ms/draw), "
          f"violations={vec_violations}, speedup={t_orig/t_vec:.1f}x")

    assert orig_violations == 0, "original sampler drew an excluded item -- test bug"
    assert vec_violations == 0, "vectorized sampler drew an excluded item -- FAILS equivalence"

    # --- Distribution equivalence: chi-square over PER-UNIQUE-ITEM frequencies ---
    # (bin by unique item, not by raw list position -- the pool is a multiset, so a duplicated item's
    # raw-list positions would otherwise inflate dof with copies of the same counter value)
    eligible_items = sorted(set(pool_items) - set(exclude_items))
    orig_freqs = np.array([orig_counter.get(i, 0) for i in eligible_items], dtype=np.float64)
    vec_freqs = np.array([vec_counter.get(i, 0) for i in eligible_items], dtype=np.float64)

    # Two-sample chi-square (contingency-table) test: orig_freqs and vec_freqs are BOTH noisy empirical
    # samples (200,000 draws each), not one fixed theoretical distribution vs. one sample -- using
    # scipy.stats.chisquare(f_obs, f_exp=orig_freqs) is the wrong test here, since it treats orig_freqs as
    # a known/deterministic expectation and so ignores its own sampling variance, understating the true
    # null-hypothesis spread and making the test far too strict (this is what produced the spurious FAIL
    # below on the first corrected run: dof=36474, stat/dof~=2.08, "significant" only because the wrong
    # variance was assumed). chi2_contingency on the 2 x n_bins table correctly accounts for variance in
    # both samples -- pool only bins with enough expected count (>=5, standard chi-square validity rule)
    # to avoid the small-expected-count degeneracy that also inflated the naive test.
    keep = (orig_freqs + vec_freqs) >= 10  # >=5 expected in each row on average
    table = np.stack([orig_freqs[keep], vec_freqs[keep]])
    stat, pvalue, dof, _ = chi2_contingency(table)
    print(f"\nTwo-sample chi-square (contingency table, {keep.sum()} bins kept of {len(eligible_items)}): "
          f"statistic={stat:.2f}, dof={dof}, p-value={pvalue:.4f}, stat/dof={stat/dof:.3f}")
    print("PASS (p > 0.01, distributions statistically indistinguishable)" if pvalue > 0.01
          else "FAIL (p <= 0.01, distributions differ significantly)")

    result = {
        "category": cat, "pool_size": len(pool_items),
        "n_freq_draws": N_FREQ_DRAWS,
        "orig_violations": orig_violations, "vec_violations": vec_violations,
        "orig_ms_per_draw": t_orig / N_FREQ_DRAWS * 1000,
        "vec_ms_per_draw": t_vec / N_FREQ_DRAWS * 1000,
        "speedup": t_orig / t_vec,
        "chi_square_statistic": float(stat), "chi_square_pvalue": float(pvalue),
        "verdict": "PASS" if (orig_violations == 0 and vec_violations == 0 and pvalue > 0.01) else "FAIL",
    }
    out_path = Path(__file__).resolve().parent.parent / "data" / "sampler_equivalence_check.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")
    assert result["verdict"] == "PASS", "Sampler equivalence check FAILED -- do not proceed to Modal runs"


if __name__ == "__main__":
    main()

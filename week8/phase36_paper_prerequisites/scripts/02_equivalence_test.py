"""
Phase 36, step 2: paired bootstrap equivalence testing on the per-query ranks
captured in step 1. Everything here is analytical -- no model is loaded.

Pre-declared (phase36_notes.md, before step 1 ran): delta = 2% relative of
ours_ens's Recall@K at each K; equivalence at K iff the 90% paired-bootstrap CI
of the difference lies within [-delta, +delta] (TOST, alpha=0.05). B=10,000,
seed 20260915. The 95% CI, an exact McNemar test on discordant hit/miss pairs,
and a per-category breakdown are reported alongside.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

B_BOOT = 10_000
SEED = 20260915
CHUNK = 500
DELTA_REL = 0.02
REFERENCE = "ours_ens"
PAIRS = [("ours_ens", "ot_ens"), ("ours_ens", "csa_ens"), ("ot_ens", "csa_ens"), ("ours_solo", "ot_solo")]
SYSTEMS = ["ours_ens", "ours_solo", "ot_ens", "ot_solo", "csa_ens"]
CROSS_SEED_VAL_STD = {  # recorded per-seed validation R@10 spreads, no new compute
    "ours (10 seeds, phase 28 individual_seeds.md)": 0.0016,
    "OutfitTransformer (3 seeds, phase 32 individual_seeds.md)": 0.00074,
    "CSA-Net random negatives (3 seeds, phase 34 individual_seeds.md)": 0.00087,
}


def load_hits(name):
    d = np.load(C.DATA_DIR / f"per_query_ranks_{name}.npz", allow_pickle=True)
    rank = d["rank"].astype(np.int64)
    assert (rank > 0).all(), f"{name}: unscored queries present"
    return {k: (rank <= k) for k in C.KS}, d["category"]


def bootstrap_means(vec, rng, n_boot=B_BOOT, chunk=CHUNK):
    """Percentile bootstrap of the mean of `vec` over resampled indices.
    Called with a fresh rng(SEED) per call so every pair/system sees the SAME
    resamples -- the paired design extends across comparisons too."""
    n = len(vec)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        bs = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(bs, n))
        out[s:s + bs] = vec[idx].mean(axis=1)
    return out


def ci(samples, level):
    lo = (1 - level) / 2 * 100
    return [float(np.percentile(samples, lo)), float(np.percentile(samples, 100 - lo))]


def main():
    hits, cats = {}, None
    for name in SYSTEMS:
        hits[name], cats = load_hits(name)
    n = len(cats)
    cat_names = sorted(set(cats.tolist()))
    print(f"{n} paired queries, {len(cat_names)} categories, B={B_BOOT}, seed={SEED}")

    ref = {k: float(hits[REFERENCE][k].mean()) for k in C.KS}
    delta = {k: DELTA_REL * ref[k] for k in C.KS}
    out = {"n_queries": n, "B": B_BOOT, "seed": SEED, "delta_rel": DELTA_REL, "reference": REFERENCE,
           "delta_abs": {str(k): delta[k] for k in C.KS}, "systems": {}, "pairs": {}, "per_category": {},
           "cross_seed_val_std_recorded": CROSS_SEED_VAL_STD}

    # per-system recall with 95% CI
    for name in SYSTEMS:
        out["systems"][name] = {}
        for k in C.KS:
            v = hits[name][k].astype(np.float64)
            bs = bootstrap_means(v, np.random.default_rng(SEED))
            out["systems"][name][str(k)] = {"recall": float(v.mean()), "ci95": ci(bs, 0.95), "se_boot": float(bs.std(ddof=1))}
        print(f"{name}: " + "  ".join(f"R@{k}={out['systems'][name][str(k)]['recall']:.4f} "
                                      f"[{out['systems'][name][str(k)]['ci95'][0]:.4f},{out['systems'][name][str(k)]['ci95'][1]:.4f}]" for k in C.KS))

    # paired differences
    for a, b in PAIRS:
        key = f"{a}__vs__{b}"
        out["pairs"][key] = {"a": a, "b": b}
        for k in C.KS:
            ha, hb = hits[a][k], hits[b][k]
            d = ha.astype(np.float64) - hb.astype(np.float64)
            bs = bootstrap_means(d, np.random.default_rng(SEED))
            point = float(d.mean())
            ci95, ci90 = ci(bs, 0.95), ci(bs, 0.90)
            equivalent = (ci90[0] >= -delta[k]) and (ci90[1] <= delta[k])
            b_disc = int((ha & ~hb).sum())
            c_disc = int((~ha & hb).sum())
            p_mcnemar = float(binomtest(b_disc, b_disc + c_disc, 0.5, alternative="two-sided").pvalue) if (b_disc + c_disc) else 1.0
            rb = float(hb.mean())
            out["pairs"][key][str(k)] = {
                "recall_a": float(ha.mean()), "recall_b": rb, "diff": point, "diff_rel_to_b": point / rb if rb else None,
                "ci95": ci95, "ci90": ci90, "delta_abs": delta[k], "tost_equivalent": bool(equivalent),
                "ci95_excludes_zero": bool(ci95[0] > 0 or ci95[1] < 0),
                "mcnemar_b_a_hit_b_miss": b_disc, "mcnemar_c_a_miss_b_hit": c_disc, "mcnemar_p_exact": p_mcnemar,
            }
            print(f"{a} - {b} @K={k}: diff={point:+.4f} ({100*point/rb:+.2f}% of {b})  CI95=[{ci95[0]:+.4f},{ci95[1]:+.4f}]"
                  f"  CI90=[{ci90[0]:+.4f},{ci90[1]:+.4f}]  delta=+/-{delta[k]:.4f}  TOST={'EQUIV' if equivalent else 'not equiv'}"
                  f"  McNemar b/c={b_disc}/{c_disc} p={p_mcnemar:.3g}")

    # per-category breakdown for the ensemble pairs
    for a, b in PAIRS[:3]:
        key = f"{a}__vs__{b}"
        out["per_category"][key] = {}
        for cat in cat_names:
            m = cats == cat
            out["per_category"][key][cat] = {"n": int(m.sum())}
            for k in C.KS:
                d = hits[a][k][m].astype(np.float64) - hits[b][k][m].astype(np.float64)
                bs = bootstrap_means(d, np.random.default_rng(SEED))
                out["per_category"][key][cat][str(k)] = {
                    "recall_a": float(hits[a][k][m].mean()), "recall_b": float(hits[b][k][m].mean()),
                    "diff": float(d.mean()), "ci95": ci(bs, 0.95)}

    with open(C.DATA_DIR / "equivalence_test.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- markdown
    L = ["# Phase 36: Equivalence Testing (paired bootstrap over test queries)", "",
         f"Per-query hit/miss captured in step 1 for all {n} test queries, identical query set for every system "
         f"(`n_skipped=0` everywhere, aggregates reproduced -- see `data/reproduction_check.json`). Paired percentile "
         f"bootstrap over query indices, B={B_BOOT}, seed {SEED}, the same resamples for every comparison. "
         f"**Pre-declared equivalence bound (phase36_notes.md): delta = {DELTA_REL:.0%} relative of `{REFERENCE}`'s Recall@K "
         f"= +/-{delta[10]:.4f} / +/-{delta[30]:.4f} / +/-{delta[50]:.4f} absolute at K=10/30/50.** Two systems are "
         f"equivalent at K iff the 90% CI of their difference lies inside [-delta, +delta] (TOST, alpha=0.05).", ""]
    L += ["## Per-system Recall@K with 95% bootstrap CI", "", "| System | R@10 [95% CI] | R@30 [95% CI] | R@50 [95% CI] |", "|---|---|---|---|"]
    for name in SYSTEMS:
        cells = []
        for k in C.KS:
            e = out["systems"][name][str(k)]
            cells.append(f"{e['recall']:.4f} [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]")
        L.append(f"| `{name}` -- {C.SYSTEM_LABELS[name]} | " + " | ".join(cells) + " |")
    L += ["", "## Paired differences (A minus B)", "",
          "| Pair | K | R@K A | R@K B | diff | diff % of B | 95% CI | 90% CI | +/-delta | TOST | McNemar b / c | McNemar p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        key = f"{a}__vs__{b}"
        for k in C.KS:
            e = out["pairs"][key][str(k)]
            L.append(f"| `{a}` - `{b}` | {k} | {e['recall_a']:.4f} | {e['recall_b']:.4f} | {e['diff']:+.4f} | "
                     f"{100*e['diff_rel_to_b']:+.2f}% | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | "
                     f"[{e['ci90'][0]:+.4f}, {e['ci90'][1]:+.4f}] | {e['delta_abs']:.4f} | "
                     f"{'**equivalent**' if e['tost_equivalent'] else 'not equivalent'} | "
                     f"{e['mcnemar_b_a_hit_b_miss']} / {e['mcnemar_c_a_miss_b_hit']} | {e['mcnemar_p_exact']:.3g} |")
    L += ["", "McNemar `b` = queries A hits and B misses; `c` = A misses and B hits; exact two-sided binomial test on the "
          "discordant pairs. `TOST` is the pre-declared equivalence verdict; the McNemar p answers the different question "
          "of whether the two systems' hit sets differ at all.", ""]
    L += ["## Recorded cross-seed variability (validation Recall@10 std, from each phase's `individual_seeds.md`)", "",
          "| System | per-seed val R@10 std |", "|---|---|"]
    for kk, v in CROSS_SEED_VAL_STD.items():
        L.append(f"| {kk} | {v} |")
    L += ["", "Query-sampling uncertainty (the bootstrap CIs above) and training-seed uncertainty (this table) are different "
          "sources; a reader needs both. The seed spreads are on the validation benchmark and were not recomputed here.", ""]
    for a, b in PAIRS[:3]:
        key = f"{a}__vs__{b}"
        L += [f"## Per-category difference, `{a}` minus `{b}` (Recall@10, 95% CI)", "",
              "| Category | n | R@10 A | R@10 B | diff | 95% CI |", "|---|---|---|---|---|---|"]
        for cat in cat_names:
            e = out["per_category"][key][cat]
            e10 = e["10"]
            L.append(f"| {cat} | {e['n']} | {e10['recall_a']:.4f} | {e10['recall_b']:.4f} | {e10['diff']:+.4f} | "
                     f"[{e10['ci95'][0]:+.4f}, {e10['ci95'][1]:+.4f}] |")
        L.append("")
    L += ["Interpretation is written in `phase36_notes.md` after reading these tables, not generated here.", ""]
    (C.PHASE_DIR / "equivalence_test.md").write_text("\n".join(L))
    print(f"saved {C.PHASE_DIR / 'equivalence_test.md'} and data/equivalence_test.json")


if __name__ == "__main__":
    main()

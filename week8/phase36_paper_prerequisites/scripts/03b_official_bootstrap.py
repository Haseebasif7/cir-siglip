"""
Phase 36, step 3b: paired bootstrap on the official-split metrics from step 3
(primary protocol). FITB correctness is a paired binary outcome over 10,000
questions -- same machinery as step 2. AUC is bootstrapped by resampling the
20,000 compatibility lines WITH REPLACEMENT and recomputing both systems' AUC on
the identical resample.

Honesty note: the pre-declared equivalence bound in phase36_notes.md was stated
for Recall@K only. The same rule (2% relative of ours_ens's value) is applied here
as an explicitly POST-HOC extension and labelled as such in every table; the CIs
and significance tests stand on their own regardless of that label.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

SEED = 20260915
B_FITB = 10_000
B_AUC = 5_000
DELTA_REL = 0.02
REFERENCE = "ours_ens"
SYSTEMS = ["ours_ens", "ours_solo", "ot_ens", "ot_solo", "csa_ens"]
PAIRS = [("ours_ens", "ot_ens"), ("ours_ens", "csa_ens"), ("ot_ens", "csa_ens"), ("ours_solo", "ot_solo")]


def load(name):
    d = np.load(C.DATA_DIR / f"official_primary_{name}.npz")
    return d["compat_scores"].astype(np.float64), d["labels"].astype(np.int64), d["fitb_correct"].astype(bool)


def boot_mean(vec, rng, n_boot, chunk=500):
    n = len(vec)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        bs = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(bs, n))
        out[s:s + bs] = vec[idx].mean(axis=1)
    return out


def boot_auc_diff(sa, sb, lab, rng, n_boot):
    n = len(lab)
    da, db = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l = lab[idx]
        da[i] = C.auc_rank_sum(sa[idx], l)
        db[i] = C.auc_rank_sum(sb[idx], l)
    return da, db


def ci(x, level):
    lo = (1 - level) / 2 * 100
    return [float(np.percentile(x, lo)), float(np.percentile(x, 100 - lo))]


def main():
    data = {n: load(n) for n in SYSTEMS}
    lab = data[REFERENCE][1]
    for n in SYSTEMS:
        assert (data[n][1] == lab).all()
    ref_auc = C.auc_rank_sum(data[REFERENCE][0], lab)
    ref_fitb = data[REFERENCE][2].mean()
    d_auc, d_fitb = DELTA_REL * ref_auc, DELTA_REL * ref_fitb
    out = {"seed": SEED, "B_fitb": B_FITB, "B_auc": B_AUC, "post_hoc_delta_rel": DELTA_REL,
           "post_hoc_delta_abs": {"auc": d_auc, "fitb": d_fitb}, "systems": {}, "pairs": {}}
    print(f"n_compat={len(lab)} n_fitb={len(data[REFERENCE][2])}; post-hoc delta: AUC +/-{d_auc:.4f}, FITB +/-{d_fitb:.4f}")

    for n in SYSTEMS:
        sa, _, fc = data[n]
        auc = C.auc_rank_sum(sa, lab)
        rng = np.random.default_rng(SEED)
        aucs = np.empty(B_AUC)
        for i in range(B_AUC):
            idx = rng.integers(0, len(lab), size=len(lab))
            aucs[i] = C.auc_rank_sum(sa[idx], lab[idx])
        fb = boot_mean(fc.astype(np.float64), np.random.default_rng(SEED), B_FITB)
        out["systems"][n] = {"auc": float(auc), "auc_ci95": ci(aucs, 0.95), "fitb": float(fc.mean()), "fitb_ci95": ci(fb, 0.95)}
        print(f"{n}: AUC {auc:.4f} {out['systems'][n]['auc_ci95']}  FITB {fc.mean():.4f} {out['systems'][n]['fitb_ci95']}")

    for a, b in PAIRS:
        key = f"{a}__vs__{b}"
        sa, _, fa = data[a]
        sb, _, fb_ = data[b]
        # AUC
        da, db = boot_auc_diff(sa, sb, lab, np.random.default_rng(SEED), B_AUC)
        diff = da - db
        pa, pb = C.auc_rank_sum(sa, lab), C.auc_rank_sum(sb, lab)
        auc_e = {"a": pa, "b": pb, "diff": float(pa - pb), "ci95": ci(diff, 0.95), "ci90": ci(diff, 0.90)}
        auc_e["ci95_excludes_zero"] = bool(auc_e["ci95"][0] > 0 or auc_e["ci95"][1] < 0)
        auc_e["post_hoc_equivalent"] = bool(auc_e["ci90"][0] >= -d_auc and auc_e["ci90"][1] <= d_auc)
        # FITB
        d = fa.astype(np.float64) - fb_.astype(np.float64)
        bs = boot_mean(d, np.random.default_rng(SEED), B_FITB)
        bd, cd = int((fa & ~fb_).sum()), int((~fa & fb_).sum())
        fitb_e = {"a": float(fa.mean()), "b": float(fb_.mean()), "diff": float(d.mean()), "ci95": ci(bs, 0.95), "ci90": ci(bs, 0.90),
                  "mcnemar_b": bd, "mcnemar_c": cd,
                  "mcnemar_p": float(binomtest(bd, bd + cd, 0.5).pvalue) if bd + cd else 1.0}
        fitb_e["ci95_excludes_zero"] = bool(fitb_e["ci95"][0] > 0 or fitb_e["ci95"][1] < 0)
        fitb_e["post_hoc_equivalent"] = bool(fitb_e["ci90"][0] >= -d_fitb and fitb_e["ci90"][1] <= d_fitb)
        out["pairs"][key] = {"a": a, "b": b, "auc": auc_e, "fitb": fitb_e}
        print(f"{a} - {b}: AUC diff {auc_e['diff']:+.4f} CI95 [{auc_e['ci95'][0]:+.4f},{auc_e['ci95'][1]:+.4f}] "
              f"sig={auc_e['ci95_excludes_zero']} postHocEq={auc_e['post_hoc_equivalent']} | "
              f"FITB diff {fitb_e['diff']:+.4f} CI95 [{fitb_e['ci95'][0]:+.4f},{fitb_e['ci95'][1]:+.4f}] "
              f"McNemar {bd}/{cd} p={fitb_e['mcnemar_p']:.3g} postHocEq={fitb_e['post_hoc_equivalent']}")

    with open(C.DATA_DIR / "official_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)

    L = ["# Phase 36: Paired Bootstrap on the Official-Split Metrics (primary protocol)", "",
         f"Seed {SEED}. FITB: paired percentile bootstrap over the 10,000 questions, B={B_FITB}, plus exact McNemar on discordant "
         f"pairs. AUC: the 20,000 compatibility lines are resampled with replacement (B={B_AUC}) and both systems' rank-sum AUC "
         f"recomputed on the identical resample. **The equivalence bound was pre-declared for Recall@K only; the same 2%-relative "
         f"rule is applied here as an explicitly post-hoc extension** (delta_AUC = +/-{d_auc:.4f}, delta_FITB = +/-{d_fitb:.4f}, "
         f"relative to `{REFERENCE}`). The CIs and significance tests do not depend on that label.", "",
         "## Per-system, with 95% bootstrap CI", "", "| System | AUC [95% CI] | FITB [95% CI] |", "|---|---|---|"]
    for n in SYSTEMS:
        e = out["systems"][n]
        L.append(f"| `{n}` | {e['auc']:.4f} [{e['auc_ci95'][0]:.4f}, {e['auc_ci95'][1]:.4f}] | "
                 f"{100*e['fitb']:.2f}% [{100*e['fitb_ci95'][0]:.2f}, {100*e['fitb_ci95'][1]:.2f}] |")
    L += ["", "## Paired differences (A minus B)", "",
          "| Pair | AUC diff | AUC 95% CI | AUC sig. (CI excl. 0) | AUC post-hoc equiv. | FITB diff (pts) | FITB 95% CI (pts) | McNemar b / c | McNemar p | FITB post-hoc equiv. |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        e = out["pairs"][f"{a}__vs__{b}"]
        au, fi = e["auc"], e["fitb"]
        L.append(f"| `{a}` - `{b}` | {au['diff']:+.4f} | [{au['ci95'][0]:+.4f}, {au['ci95'][1]:+.4f}] | {'yes' if au['ci95_excludes_zero'] else 'no'} | "
                 f"{'yes' if au['post_hoc_equivalent'] else 'no'} | {100*fi['diff']:+.2f} | [{100*fi['ci95'][0]:+.2f}, {100*fi['ci95'][1]:+.2f}] | "
                 f"{fi['mcnemar_b']} / {fi['mcnemar_c']} | {fi['mcnemar_p']:.3g} | {'yes' if fi['post_hoc_equivalent'] else 'no'} |")
    L += ["", "Interpretation in `phase36_notes.md`.", ""]
    (C.PHASE_DIR / "official_bootstrap.md").write_text("\n".join(L))
    print("saved official_bootstrap.md and data/official_bootstrap.json")


if __name__ == "__main__":
    main()

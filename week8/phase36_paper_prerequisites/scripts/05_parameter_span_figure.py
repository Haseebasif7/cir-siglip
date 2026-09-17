"""
Phase 36, step 5: the parameter-span figure -- trainable parameters per member
(log x) against test Recall@K (y, from zero) for the three matched-condition
mechanisms, filled = final ensemble, hollow = single model, with the untrained
SigLIP baseline as a dashed reference. Error bars = 95% paired-bootstrap CIs from
step 2. Built per the dataviz skill: form chosen for identity+magnitude (dot per
system, small multiples over K, one axis), fixed-order categorical hues validated
with the skill's script (light mode: all checks pass; aqua carries a contrast WARN,
so every point is direct-labelled and marker shape is a secondary encoding), thin
marks, recessive grid, text in ink tokens not series colors.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"ours": "#2a78d6", "ot": "#eb6834", "csa": "#1baf7a"}  # fixed categorical order, validated
MARK = {"ours": "o", "ot": "s", "csa": "^"}
LABEL = {"ours": "Ours (mean-pooled projection)", "ot": "OutfitTransformer mechanism", "csa": "CSA-Net mechanism"}
RAW_SIGLIP_TEST = {10: 0.0553, 30: 0.1067, 50: 0.1437}  # phase 19, re-measured on this test benchmark
SYSTEMS = [  # name, kind, members
    ("ours_ens", "ours", 10), ("ours_solo", "ours", 1),
    ("ot_ens", "ot", 3), ("ot_solo", "ot", 1),
    ("csa_ens", "csa", 3),
]


def main():
    eq = json.load(open(C.DATA_DIR / "equivalence_test.json"))["systems"]
    rows = []
    for name, kind, members in SYSTEMS:
        for k in C.KS:
            e = eq[name][str(k)]
            rows.append({"system": name, "kind": kind, "members": members, "params_per_member": C.N_PARAMS[kind],
                         "params_total": C.N_PARAMS[kind] * members, "K": k, "recall": e["recall"],
                         "ci95_lo": e["ci95"][0], "ci95_hi": e["ci95"][1]})
    with open(C.DATA_DIR / "parameter_span.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=False, constrained_layout=True)
    for ax, k in zip(axes, C.KS):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axhline(RAW_SIGLIP_TEST[k], color=INK2, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
        ax.text(6.2e4, RAW_SIGLIP_TEST[k] + 0.006, "untrained SigLIP", color=INK2, fontsize=6.8, va="bottom", ha="left")
        for name, kind, members in SYSTEMS:
            e = eq[name][str(k)]
            x, y = C.N_PARAMS[kind], e["recall"]
            ens = members > 1
            ax.errorbar(x, y, yerr=[[y - e["ci95"][0]], [e["ci95"][1] - y]], fmt="none", ecolor=COL[kind], elinewidth=0.9, capsize=2, zorder=2)
            ax.plot(x, y, marker=MARK[kind], markersize=6.5 if ens else 5.2, color=COL[kind],
                    markerfacecolor=COL[kind] if ens else "white", markeredgewidth=1.3, linestyle="none", zorder=3)
        # direct labels (ensembles), placed to avoid the ours/OT collision on the log axis
        if k == 10:
            ax.annotate("CSA-Net mech.\n3 seeds", (C.N_PARAMS["csa"], eq["csa_ens"]["10"]["recall"]), xytext=(4, -9),
                        textcoords="offset points", ha="left", va="top", fontsize=6.8, color=INK)
            ax.annotate("OutfitTransformer mech.\n3 seeds", (C.N_PARAMS["ot"], eq["ot_ens"]["10"]["recall"]), xytext=(-4, 9),
                        textcoords="offset points", ha="right", va="bottom", fontsize=6.8, color=INK)
            ax.annotate("Ours\n10 seeds", (C.N_PARAMS["ours"], eq["ours_ens"]["10"]["recall"]), xytext=(6, 9),
                        textcoords="offset points", ha="left", va="bottom", fontsize=6.8, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(5e4, 4.5e6)
        ax.set_xticks([1e5, 1e6])
        ax.set_xticklabels(["100 k", "1 M"])
        ymax = max(eq[n][str(k)]["ci95"][1] for n, _, _ in SYSTEMS)
        ax.set_ylim(0, ymax * 1.28)
        ax.set_title(f"Recall@{k}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("trainable parameters per member (log)", fontsize=7.5, color=INK2)
        ax.tick_params(length=2.5, width=0.5, labelsize=7)
    axes[0].set_ylabel("test Recall@K", fontsize=7.5, color=INK2)
    handles = [Line2D([], [], marker=MARK[kd], color=COL[kd], linestyle="none", markersize=6, label=LABEL[kd]) for kd in ("ours", "ot", "csa")]
    handles += [Line2D([], [], marker="o", color=INK2, markerfacecolor=INK2, linestyle="none", markersize=6, label="final ensemble"),
                Line2D([], [], marker="o", color=INK2, markerfacecolor="white", linestyle="none", markersize=5, label="single model")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.08), handletextpad=0.4, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG_DIR / f"parameter_span.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved figures/parameter_span.png/.pdf and data/parameter_span.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()

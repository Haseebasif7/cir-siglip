"""
Phase 12c, step 4.2: the full control-effectiveness diagnostic, phases 12 and
12b's numbers included for a three-way comparison, including the decisive
substitute-vs-raw-SigLIP / complement-vs-raw-SigLIP overlap check phase 12b
added -- this is what step 5's stopping condition is judged against.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import load_benchmark, topk_for_query
from model import ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT = BASE_DIR / "models" / "ranking_distillation.pt"
OUT_MD = BASE_DIR / "control_effectiveness.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 10
SAMPLE_SIZE = 1000
OVERLAP_SAMPLE_SIZE = 500
SEED = 42

# From phase 12's and phase 12b's own control_effectiveness.md.
PHASE12 = {"axis1_sub": 0.7193, "axis1_comp": 0.7120, "axis2_sub": 0.1280, "axis2_comp": 0.1280,
           "overlap_modes": 0.7636, "per_item_cos": 0.8288, "sub_vs_raw": None, "comp_vs_raw": None}
PHASE12B = {"axis1_sub": 0.7233, "axis1_comp": 0.7228, "axis2_sub": 0.1250, "axis2_comp": 0.1180,
            "overlap_modes": 0.5102, "per_item_cos": 0.5844, "sub_vs_raw": 0.1612, "comp_vs_raw": 0.1754}
# STOPPING_CONDITION_THRESHOLD: the brief calls for "a real, clear gap", not a
# small/noise-level one, between substitute-vs-raw overlap and complement-vs-raw
# overlap. 0.03 (3 percentage points on a top-10 overlap fraction, i.e. roughly
# 0.3 items out of 10) is used as a concrete, stated-in-advance threshold for
# "real" vs "noise-level" -- chosen because phase 12b's own reversal on axis 2
# (0.007, well under this) was judged noise-level, so the bar for "real" here is
# set clearly above that.
STOPPING_CONDITION_THRESHOLD = 0.03


def load_raw_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


@torch.no_grad()
def project_all(model, embeddings, alpha, batch_size=4096):
    outs = []
    for start in range(0, len(embeddings), batch_size):
        x = torch.tensor(embeddings[start:start + batch_size], device=DEVICE)
        outs.append(model(x, alpha=alpha).cpu().numpy())
    return np.concatenate(outs, axis=0)


def cmp3(name, new, p12, p12b, fmt="{:.4f}"):
    p12_s = fmt.format(p12) if p12 is not None else "--"
    p12b_s = fmt.format(p12b) if p12b is not None else "--"
    return f"- {name}: phase 12 = {p12_s}, phase 12b = {p12b_s}, **phase 12c = {fmt.format(new)}**"


def main():
    pools, queries = load_benchmark()
    item_ids, raw_emb = load_raw_embeddings()
    idx = {a: i for i, a in enumerate(item_ids)}

    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    sub_emb = project_all(model, raw_emb, alpha=1.0)
    comp_emb = project_all(model, raw_emb, alpha=0.0)

    # --- Axes 1 & 2 ---
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(queries), size=min(SAMPLE_SIZE, len(queries)), replace=False)

    def query_raw_vec(query_items):
        item_idx = [idx[i] for i in query_items if i in idx]
        if not item_idx:
            return None
        v = raw_emb[item_idx].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    sub_avg_raw_sims, comp_avg_raw_sims = [], []
    sub_hits, comp_hits = 0, 0
    n_scored = 0
    for qi in sample_idx:
        q = queries[qi]
        qrv = query_raw_vec(q["query_items"])
        if qrv is None:
            continue
        sub_top, _ = topk_for_query(pools, q, item_ids, sub_emb, k=TOP_K)
        comp_top, _ = topk_for_query(pools, q, item_ids, comp_emb, k=TOP_K)
        if not sub_top or not comp_top:
            continue
        sub_avg_raw_sims.append(float(np.mean(raw_emb[[idx[i] for i in sub_top]] @ qrv)))
        comp_avg_raw_sims.append(float(np.mean(raw_emb[[idx[i] for i in comp_top]] @ qrv)))
        sub_hits += int(q["target_item"] in sub_top)
        comp_hits += int(q["target_item"] in comp_top)
        n_scored += 1

    sub_avg = float(np.mean(sub_avg_raw_sims))
    comp_avg = float(np.mean(comp_avg_raw_sims))
    sub_hit_rate = sub_hits / n_scored
    comp_hit_rate = comp_hits / n_scored

    # --- Overlap between modes + per-item cosine ---
    overlap_rng = np.random.default_rng(SEED)
    overlap_sample = overlap_rng.choice(len(queries), size=OVERLAP_SAMPLE_SIZE, replace=False)
    overlaps_modes, overlaps_sub_raw, overlaps_comp_raw = [], [], []
    for qi in overlap_sample:
        q = queries[qi]
        sub_top, _ = topk_for_query(pools, q, item_ids, sub_emb, k=TOP_K)
        comp_top, _ = topk_for_query(pools, q, item_ids, comp_emb, k=TOP_K)
        raw_top, _ = topk_for_query(pools, q, item_ids, raw_emb, k=TOP_K)
        if not sub_top or not comp_top or not raw_top:
            continue
        overlaps_modes.append(len(set(sub_top) & set(comp_top)) / TOP_K)
        overlaps_sub_raw.append(len(set(sub_top) & set(raw_top)) / TOP_K)
        overlaps_comp_raw.append(len(set(comp_top) & set(raw_top)) / TOP_K)

    overlap_modes_mean = float(np.mean(overlaps_modes))
    sub_vs_raw_mean = float(np.mean(overlaps_sub_raw))
    comp_vs_raw_mean = float(np.mean(overlaps_comp_raw))

    per_item_cos_sample = raw_emb[:5000]
    with torch.no_grad():
        x = torch.tensor(per_item_cos_sample, device=DEVICE)
        z_sub_s = model(x, alpha=1.0).cpu().numpy()
        z_comp_s = model(x, alpha=0.0).cpu().numpy()
    per_item_cos = float(np.mean(np.sum(z_sub_s * z_comp_s, axis=1)))

    gap = sub_vs_raw_mean - comp_vs_raw_mean
    stopping_condition_triggered = gap < STOPPING_CONDITION_THRESHOLD

    lines = [
        "# Phase 12c, Step 4.2: Control-Effectiveness Diagnostic, Three-Way Comparison",
        "",
        f"Same procedure as phases 12/12b (sample size {SAMPLE_SIZE}, seed={SEED}, top-{TOP_K}), "
        "on the ranking-distillation checkpoint.",
        "",
        "## Axis 1: visual similarity to the query (raw SigLIP cosine, top-10 average)",
        "",
        cmp3("Substitute mode", sub_avg, PHASE12["axis1_sub"], PHASE12B["axis1_sub"]),
        cmp3("Complement mode", comp_avg, PHASE12["axis1_comp"], PHASE12B["axis1_comp"]),
        f"- Gap (substitute - complement): phase 12 = {PHASE12['axis1_sub']-PHASE12['axis1_comp']:.4f}, "
        f"phase 12b = {PHASE12B['axis1_sub']-PHASE12B['axis1_comp']:.4f}, "
        f"**phase 12c = {sub_avg-comp_avg:.4f}**",
        "",
        "## Axis 2: match rate against real outfit co-occurrence (hit@10 on the true target)",
        "",
        cmp3("Substitute mode", sub_hit_rate, PHASE12["axis2_sub"], PHASE12B["axis2_sub"]),
        cmp3("Complement mode", comp_hit_rate, PHASE12["axis2_comp"], PHASE12B["axis2_comp"]),
        "",
        "## Overlap and per-item cosine between the two modes",
        "",
        cmp3(f"Top-10 overlap between modes (n={len(overlaps_modes)})", overlap_modes_mean,
             PHASE12["overlap_modes"], PHASE12B["overlap_modes"]),
        cmp3("Per-item cosine(z_sub, z_comp), 5,000 items", per_item_cos,
             PHASE12["per_item_cos"], PHASE12B["per_item_cos"]),
        "",
        "## THE DECISIVE CHECK: does substitute mode's retrieval now resemble raw SigLIP's "
        "more than complement mode's does?",
        "",
        cmp3("Substitute mode vs raw SigLIP top-10 overlap", sub_vs_raw_mean,
             PHASE12["sub_vs_raw"], PHASE12B["sub_vs_raw"]),
        cmp3("Complement mode vs raw SigLIP top-10 overlap", comp_vs_raw_mean,
             PHASE12["comp_vs_raw"], PHASE12B["comp_vs_raw"]),
        f"- **Gap (substitute vs raw) - (complement vs raw) = {gap:.4f}** "
        f"(phase 12b's gap was {PHASE12B['sub_vs_raw']-PHASE12B['comp_vs_raw']:.4f}, in the "
        "WRONG direction -- substitute was slightly LOWER).",
        f"- Stopping-condition threshold (stated in advance, see script docstring): a gap "
        f"below {STOPPING_CONDITION_THRESHOLD:.2f} counts as noise-level, not a real "
        "separation.",
        f"- **{'STOPPING CONDITION TRIGGERED' if stopping_condition_triggered else 'GAP CLEARS THE THRESHOLD'}**: "
        f"gap = {gap:.4f} {'<' if stopping_condition_triggered else '>='} {STOPPING_CONDITION_THRESHOLD:.2f}.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT_MD}")
    print(f"\nSTOPPING_CONDITION_TRIGGERED={stopping_condition_triggered}, gap={gap:.4f}")


if __name__ == "__main__":
    main()

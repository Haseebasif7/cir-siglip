"""
Phase 12b, step 4.2: re-run phase 12's exact control-effectiveness diagnostic
(same sample size/seed, same two axes, same overlap/per-item-cosine follow-up
checks that actually exposed the original failure) on the newly retrained
checkpoint, with phase 12's own numbers carried forward for direct
before/after comparison. This -- not Recall@K -- is the real bar the fix
needs to clear (see phase 12's own control_effectiveness.md).
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
CHECKPOINT = BASE_DIR / "models" / "controllable_modes_fixed.pt"
OUT_MD = BASE_DIR / "control_effectiveness.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 10
SAMPLE_SIZE = 1000
OVERLAP_SAMPLE_SIZE = 500
SEED = 42

# From week4/phase12_controllable_modes/control_effectiveness.md (phase 12's own run).
PHASE12_AXIS1 = {"substitute": 0.7193, "complement": 0.7120}
PHASE12_AXIS2 = {"substitute": 0.1280, "complement": 0.1280}
PHASE12_OVERLAP_MEAN = 0.7636
PHASE12_PER_ITEM_COSINE = 0.8288


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


def main():
    pools, queries = load_benchmark()
    item_ids, raw_emb = load_raw_embeddings()
    idx = {a: i for i, a in enumerate(item_ids)}

    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    sub_emb = project_all(model, raw_emb, alpha=1.0)
    comp_emb = project_all(model, raw_emb, alpha=0.0)

    # --- Axes 1 & 2, same sample/seed as phase 12 ---
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
    direction_visual_ok = sub_avg > comp_avg
    direction_cooccur_ok = comp_hit_rate > sub_hit_rate

    # --- Follow-up: top-10 overlap + per-item cosine, same as phase 12 ---
    overlap_rng = np.random.default_rng(SEED)
    overlap_sample = overlap_rng.choice(len(queries), size=OVERLAP_SAMPLE_SIZE, replace=False)
    overlaps = []
    for qi in overlap_sample:
        q = queries[qi]
        sub_top, _ = topk_for_query(pools, q, item_ids, sub_emb, k=TOP_K)
        comp_top, _ = topk_for_query(pools, q, item_ids, comp_emb, k=TOP_K)
        if not sub_top or not comp_top:
            continue
        overlaps.append(len(set(sub_top) & set(comp_top)) / TOP_K)
    overlap_mean = float(np.mean(overlaps))

    per_item_cos_sample = raw_emb[:5000]
    with torch.no_grad():
        x = torch.tensor(per_item_cos_sample, device=DEVICE)
        z_sub_s = model(x, alpha=1.0).cpu().numpy()
        z_comp_s = model(x, alpha=0.0).cpu().numpy()
    per_item_cos = float(np.mean(np.sum(z_sub_s * z_comp_s, axis=1)))

    def cmp_line(name, new, old):
        d = new - old
        sign = "+" if d >= 0 else ""
        return f"- {name}: phase 12 = {old:.4f}, **phase 12b = {new:.4f}** (delta {sign}{d:.4f})"

    lines = [
        "# Phase 12b, Step 4.2: Control-Effectiveness Diagnostic, Before/After Comparison",
        "",
        f"Same procedure as phase 12's `control_effectiveness.md` (sample size {SAMPLE_SIZE}, "
        f"seed={SEED}, top-{TOP_K}), on the retrained `controllable_modes_fixed.pt` checkpoint.",
        "",
        "## Axis 1: visual similarity to the query (raw SigLIP cosine, top-10 average)",
        "",
        cmp_line("Substitute mode", sub_avg, PHASE12_AXIS1["substitute"]),
        cmp_line("Complement mode", comp_avg, PHASE12_AXIS1["complement"]),
        f"- **Direction {'CONFIRMED' if direction_visual_ok else 'NOT CONFIRMED'}** "
        f"(substitute {'>' if direction_visual_ok else '<='} complement); gap = "
        f"{sub_avg - comp_avg:.4f} (phase 12's gap was {PHASE12_AXIS1['substitute']-PHASE12_AXIS1['complement']:.4f}).",
        "",
        "## Axis 2: match rate against real outfit co-occurrence (hit@10 on the true target)",
        "",
        cmp_line("Substitute mode", sub_hit_rate, PHASE12_AXIS2["substitute"]),
        cmp_line("Complement mode", comp_hit_rate, PHASE12_AXIS2["complement"]),
        f"- **Direction {'CONFIRMED' if direction_cooccur_ok else 'NOT CONFIRMED'}** "
        f"(complement {'>' if direction_cooccur_ok else '<='} substitute); phase 12 was an exact "
        f"tie (128/1000 each) -- {'this tie is now broken.' if sub_hits != comp_hits else 'the tie is NOT broken.'}",
        "",
        "## Follow-up checks (these are what actually exposed the problem in phase 12)",
        "",
        cmp_line(f"Mean top-10 overlap between the two modes (n={len(overlaps)} queries)",
                 overlap_mean, PHASE12_OVERLAP_MEAN),
        cmp_line("Mean per-item cosine(z_sub, z_comp) across 5,000 items",
                 per_item_cos, PHASE12_PER_ITEM_COSINE),
        "",
        "## Verdict",
        "",
    ]

    both_confirmed = direction_visual_ok and direction_cooccur_ok
    overlap_dropped_meaningfully = (PHASE12_OVERLAP_MEAN - overlap_mean) >= 0.15
    tie_broken = sub_hits != comp_hits

    if both_confirmed and overlap_dropped_meaningfully and tie_broken:
        lines.append("**Fix confirmed working on every diagnostic axis.** Both premises hold, the "
                      "axis-2 exact tie is broken, and top-10 overlap dropped meaningfully from "
                      "phase 12's 76.4%. See `../phase12b_notes.md` for the full go/no-go verdict.")
    elif overlap_dropped_meaningfully or tie_broken or both_confirmed:
        lines.append("**Partial improvement.** Some diagnostics moved in the right direction, others "
                      "did not fully separate. See `../phase12b_notes.md` for the full go/no-go "
                      "verdict and what this means for whether to pursue the fuller architecture.")
    else:
        lines.append("**Fix did not meaningfully change the underlying problem.** The two modes "
                      "still barely separate on the diagnostics that matter (overlap, per-item "
                      "cosine, axis 2). See `../phase12b_notes.md` for the full go/no-go verdict.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()

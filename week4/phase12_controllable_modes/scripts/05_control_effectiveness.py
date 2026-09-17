"""
Phase 12, step 4.2: does the control knob actually do what it claims?

For a sample of CIR benchmark queries, compare substitute mode's and
complement mode's top-10 results directly on two axes:

  1. Average raw SigLIP similarity between the query and its top-10 results
     -- substitute mode's results should be measurably MORE visually similar
     to the query than complement mode's (that's the entire premise of
     "substitute mode should behave like plain visual similarity").
  2. Match rate against real outfit co-occurrence ground truth -- for each
     query, whether the true held-out target (a real complement, by
     definition of how the CIR benchmark's leave-one-out queries were built)
     appears in the top-10 -- complement mode's hit rate should be higher
     than substitute mode's.

This is the direct mechanism check: the CIR Recall@K numbers in
results_table.md show aggregate retrieval quality, but say nothing on their
own about whether the two modes are actually behaving differently in the
expected direction, or just producing similar rankings with different labels.
"""
import json
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
CHECKPOINT = BASE_DIR / "models" / "controllable_modes.pt"
OUT_MD = BASE_DIR / "control_effectiveness.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 10
SAMPLE_SIZE = 1000
SEED = 42


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

        sub_top_raw_idx = [idx[i] for i in sub_top]
        comp_top_raw_idx = [idx[i] for i in comp_top]
        sub_avg_raw_sims.append(float(np.mean(raw_emb[sub_top_raw_idx] @ qrv)))
        comp_avg_raw_sims.append(float(np.mean(raw_emb[comp_top_raw_idx] @ qrv)))

        sub_hits += int(q["target_item"] in sub_top)
        comp_hits += int(q["target_item"] in comp_top)
        n_scored += 1

    sub_avg = float(np.mean(sub_avg_raw_sims))
    comp_avg = float(np.mean(comp_avg_raw_sims))
    sub_hit_rate = sub_hits / n_scored
    comp_hit_rate = comp_hits / n_scored

    direction_visual_ok = sub_avg > comp_avg
    direction_cooccur_ok = comp_hit_rate > sub_hit_rate

    lines = [
        "# Phase 12, Step 4.2: Control-Effectiveness Diagnostic",
        "",
        f"Sample: {n_scored} CIR benchmark queries (seed={SEED}, target sample size {SAMPLE_SIZE}), "
        f"top-{TOP_K} results compared under substitute mode (alpha=1.0) and complement mode (alpha=0.0), "
        "same shared checkpoint, same queries.",
        "",
        "## Axis 1: visual similarity to the query (raw SigLIP cosine, averaged over top-10 results)",
        "",
        "Premise: substitute mode should look MORE like plain visual similarity, so its top-10 results "
        "should have higher average raw-SigLIP similarity to the query than complement mode's.",
        "",
        f"- Substitute mode: {sub_avg:.4f}",
        f"- Complement mode: {comp_avg:.4f}",
        f"- **Direction {'CONFIRMED' if direction_visual_ok else 'NOT CONFIRMED'}** "
        f"(substitute {'>' if direction_visual_ok else '<='} complement).",
        "",
        "## Axis 2: match rate against real outfit co-occurrence ground truth (hit@10 on the true held-out target)",
        "",
        "Premise: complement mode should better match real 'goes well with' outfit assembly, so its hit "
        "rate against the CIR benchmark's true target (a real Polyvore co-outfit partner, by construction) "
        "should be higher than substitute mode's.",
        "",
        f"- Substitute mode: {sub_hit_rate:.4f} ({sub_hits}/{n_scored})",
        f"- Complement mode: {comp_hit_rate:.4f} ({comp_hits}/{n_scored})",
        f"- **Direction {'CONFIRMED' if direction_cooccur_ok else 'NOT CONFIRMED'}** "
        f"(complement {'>' if direction_cooccur_ok else '<='} substitute).",
        "",
        "## Verdict",
        "",
    ]
    if direction_visual_ok and direction_cooccur_ok:
        lines.append("**Both directions confirmed** -- the control knob measurably changes retrieval "
                      "behavior in the intended direction on both axes, not just in aggregate Recall@K "
                      "numbers. This is real evidence the mode mechanism works as designed, not an "
                      "artifact of the two modes producing near-identical rankings.")
    elif direction_visual_ok or direction_cooccur_ok:
        lines.append("**Only one direction confirmed.** The control knob moves behavior as intended on "
                      "one axis but not the other -- see phase12_notes.md for interpretation of which "
                      "axis failed and why that matters for the overall recommendation.")
    else:
        lines.append("**Neither direction confirmed.** The control knob does not measurably separate "
                      "the two modes' behavior on either axis tested -- a significant negative finding "
                      "for this mechanism, see phase12_notes.md.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()

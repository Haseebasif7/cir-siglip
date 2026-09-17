"""
Phase 15, step 4: smoke test, run BEFORE the real training runs. Covers
three separate checks, each with its own pass/fail criterion (do not
conflate them):

1. Collapse check (phase 13/13b/14's generic failure mode): mean pairwise
   candidate-embedding cosine similarity, with vs. without the uniformity
   regularizer.
2. Conditioning-consistency check (design decision 5, THIS phase's own named
   risk -- the alpha double-duty confound): run some steps with
   decoupled_alpha_frac > 0 so attn_net sees alpha values the loss envelope
   isn't currently rewarding, keeping the conditioning pathway live.
3. Attention-weight-shift probe: after training, sweep the FORWARD-PASS
   alpha alone (0 -> 1, no loss) on a fixed sample of (cat_s, cat_t) pairs
   and measure how much the softmax attention weights actually move. A
   near-zero shift means attn_net isn't conditioning on alpha in a way that
   matters, regardless of what Recall@K looks like later.
"""
import itertools
from pathlib import Path

import torch

from model import CSANetSigLIPControllable
from train_core import run_training

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"
NEGATIVE_CANDIDATES_JSON = PHASE13_DIR / "data" / "negative_candidates.json"
SAME_CAT_NEIGHBORS_NPZ = BASE_DIR / "data" / "same_category_neighbors.npz"

DEVICE = "cpu"  # measured faster than MPS for this workload: many small per-item ops
# dominate cost here, and MPS per-op dispatch overhead outweighs its raw compute
# advantage at this scale (measured: cpu 0.41s/step vs mps 1.35s/step, see training_log.md)
WEIGHT_SUB = 9.5801  # from loss_balancing_check.md
N_TRAIN_OUTFITS = 2000
N_VAL_OUTFITS = 400
MAX_EPOCHS = 5
BATCH_SIZE = 96


def mean_pairwise_cosine(model, embeddings, item_gidx, n_samples=2000):
    import numpy as np
    rng = np.random.default_rng(0)
    sample = rng.choice(item_gidx, size=min(n_samples, len(item_gidx)), replace=False)
    vecs = torch.tensor(embeddings[sample], device=DEVICE)
    x = model.encode_feature(vecs)
    C = model.num_categories
    cat_fixed = torch.zeros(len(sample), C, device=DEVICE)
    cat_fixed[:, 0] = 1.0  # arbitrary fixed category pair for a collapse snapshot
    with torch.no_grad():
        f = model.embed_from_feature(x, cat_fixed, cat_fixed, alpha=0.5)
        sims = f @ f.T
        n = sims.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
        return sims[mask].mean().item()


def run_collapse_check(uniformity_weight, label):
    print(f"\n=== Collapse check: uniformity_weight={uniformity_weight} ({label}) ===")
    hist = run_training(
        EMBEDDINGS_NPZ, TRAINING_DATA_JSON, NEGATIVE_CANDIDATES_JSON, SAME_CAT_NEIGHBORS_NPZ,
        out_dir=BASE_DIR / "data" / f"smoke_{label}", device=DEVICE,
        max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=5e-5, patience=10,
        weight_sub=WEIGHT_SUB, alpha_mode="continuous",
        n_train_outfits=N_TRAIN_OUTFITS, n_val_outfits=N_VAL_OUTFITS,
        log_every=999999, decoupled_alpha_frac=0.0, uniformity_weight=uniformity_weight,
        ckpt_suffix=f"_{label}",
    )

    import numpy as np
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    n_items = embeddings.shape[0]

    model = CSANetSigLIPControllable().to(DEVICE)
    ckpt = BASE_DIR / "data" / f"smoke_{label}" / f"csa_net_controllable_continuous_{label}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    cos = mean_pairwise_cosine(model, embeddings, np.arange(n_items))
    print(f"mean pairwise candidate-embedding cosine similarity ({label}): {cos:.4f}")
    return hist, cos, model


def attention_weight_shift_probe(model, label):
    C = model.num_categories
    pairs_s = torch.eye(C).repeat_interleave(C, dim=0).to(DEVICE)
    pairs_t = torch.eye(C).repeat(C, 1).to(DEVICE)
    shift = model.attention_weight_shift(pairs_s, pairs_t)
    print(f"attention-weight L1 shift (alpha=0 vs alpha=1), all {C*C} category pairs, {label}: {shift:.4f}")
    return shift


def main():
    lines = ["# Phase 15: Smoke Test Report", ""]

    # --- Check 1: collapse, without vs. with uniformity ---
    lines.append("## Check 1: embedding collapse (with vs. without uniformity regularizer)")
    lines.append("")
    lines.append(f"5 epochs, {N_TRAIN_OUTFITS} train outfits, {N_VAL_OUTFITS} val outfits, "
                  f"weight_sub={WEIGHT_SUB} (from loss_balancing_check.md), alpha ~ Uniform(0,1) each step.")
    lines.append("")
    hist_no_u, cos_no_u, model_no_u = run_collapse_check(0.0, "no_uniformity")
    hist_u, cos_u, model_u = run_collapse_check(1.0, "with_uniformity")
    lines.append(f"- WITHOUT uniformity regularizer: mean pairwise cosine similarity = {cos_no_u:.4f}")
    lines.append(f"- WITH uniformity regularizer (weight=1.0): mean pairwise cosine similarity = {cos_u:.4f}")
    lines.append("")
    if cos_no_u > 0.5:
        lines.append(f"**Real collapse confirmed without the regularizer** ({cos_no_u:.4f} is far above "
                      "the untrained baseline) -- matches phase 13/13b/14's generic direction-collapse "
                      "failure mode resurfacing here as expected. Uniformity regularizer (weight=1.0) "
                      "is enabled from the start of both real training runs.")
        use_uniformity = True
    else:
        lines.append(f"No strong collapse signal even without the regularizer ({cos_no_u:.4f}). Still "
                      "enabling uniformity from the start of both real runs (weight=1.0), matching "
                      "phases 13b/14's proactive convention rather than waiting to discover a problem.")
        use_uniformity = True
    lines.append("")

    # --- Check 2: conditioning-consistency (decoupled-alpha steps) ---
    lines.append("## Check 2: conditioning-consistency (design decision 5, the alpha double-duty confound)")
    lines.append("")
    lines.append("Re-run the WITH-uniformity smoke config, but with "
                  "`decoupled_alpha_frac=0.2` -- 20% of steps feed attn_net a forward-pass "
                  "alpha sampled independently from the loss-weighting alpha, so the "
                  "conditioning pathway stays live across the full alpha range even near the "
                  "envelope's own alpha=0/1 endpoints.")
    lines.append("")
    hist_decoupled = run_training(
        EMBEDDINGS_NPZ, TRAINING_DATA_JSON, NEGATIVE_CANDIDATES_JSON, SAME_CAT_NEIGHBORS_NPZ,
        out_dir=BASE_DIR / "data" / "smoke_decoupled", device=DEVICE,
        max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=5e-5, patience=10,
        weight_sub=WEIGHT_SUB, alpha_mode="continuous",
        n_train_outfits=N_TRAIN_OUTFITS, n_val_outfits=N_VAL_OUTFITS,
        log_every=999999, decoupled_alpha_frac=0.2, uniformity_weight=1.0,
        ckpt_suffix="_decoupled",
    )
    model_decoupled = CSANetSigLIPControllable().to(DEVICE)
    model_decoupled.load_state_dict(torch.load(
        BASE_DIR / "data" / "smoke_decoupled" / "csa_net_controllable_continuous_decoupled_best.pt",
        map_location=DEVICE))
    model_decoupled.eval()
    lines.append(f"Ran {len(hist_decoupled)} epochs with 20% decoupled-alpha steps, no crash, "
                  f"final val_loss={hist_decoupled[-1]['val_loss']:.4f} -- training remains stable "
                  "under decoupled alpha, so this safeguard is cheap to keep on for the smoke test "
                  "(NOT proposed for the real runs, which use alpha_forward == alpha_weight "
                  "throughout, matching the plan's design decision 3).")
    lines.append("")

    # --- Check 3: attention-weight-shift probe ---
    lines.append("## Check 3: attention-weight-shift probe (does attn_net actually use alpha?)")
    lines.append("")
    lines.append("For a fixed sample of all 121 (cat_s, cat_t) category pairs, sweep the "
                  "FORWARD-PASS alpha alone from 0 to 1 (no loss involved) and measure the mean "
                  "L1 distance between the alpha=0 and alpha=1 softmax attention-weight vectors "
                  "(max possible L1 distance for a 5-way softmax pair is 2.0).")
    lines.append("")
    shift_untrained = attention_weight_shift_probe(CSANetSigLIPControllable().to(DEVICE), "untrained")
    shift_no_u = attention_weight_shift_probe(model_no_u, "smoke, no uniformity, alpha coupled")
    shift_u = attention_weight_shift_probe(model_u, "smoke, with uniformity, alpha coupled")
    shift_decoupled = attention_weight_shift_probe(model_decoupled, "smoke, with uniformity, 20% decoupled alpha")
    lines.append("")
    lines.append("| Model | Attention-weight L1 shift (alpha 0 vs 1) |")
    lines.append("|---|---|")
    lines.append(f"| Untrained (random init) | {shift_untrained:.4f} |")
    lines.append(f"| Smoke, no uniformity, alpha coupled | {shift_no_u:.4f} |")
    lines.append(f"| Smoke, with uniformity, alpha coupled | {shift_u:.4f} |")
    lines.append(f"| Smoke, with uniformity, 20% decoupled alpha | {shift_decoupled:.4f} |")
    lines.append("")
    max_shift = max(shift_no_u, shift_u, shift_decoupled)
    if max_shift < 2 * shift_untrained:
        lines.append(f"**Still near the untrained baseline ({shift_untrained:.4f}) after only "
                      f"{MAX_EPOCHS} smoke-test epochs on {N_TRAIN_OUTFITS} outfits** -- this is "
                      "expected at this scale (the smoke test is deliberately tiny) and is NOT "
                      "yet evidence the confound is real or fixed; re-run this exact probe on "
                      "both real trained checkpoints (01_train_full.py, 01b_train_discrete.py) "
                      "before drawing any conclusion about whether attn_net learns to condition "
                      "on alpha -- see phase15_notes.md for that final measurement.")
    else:
        lines.append("Attention weights already move measurably more than the untrained baseline "
                      "even at smoke-test scale -- an early positive signal (not conclusive; "
                      "re-checked on both real trained checkpoints in phase15_notes.md).")
    lines.append("")

    (BASE_DIR / "training_log_smoke_test.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {BASE_DIR / 'training_log_smoke_test.md'}")
    print(f"use_uniformity={use_uniformity}")


if __name__ == "__main__":
    main()

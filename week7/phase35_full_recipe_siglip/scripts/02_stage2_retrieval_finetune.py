"""
Phase 35, stage 2: retrieval fine-tuning with target-category conditioning
(piece 2) and curriculum negative sampling (piece 3), warm-started from
stage 1's CP pre-training checkpoint (piece 1).

Hyperparameters are phase 31/32's exact winning configuration, held
UNCHANGED -- this is what makes the phase32-vs-phase35 comparison a clean
isolation of the training recipe's contribution alone (see model.py's
module docstring and ../implementation_notes.md). The only new
hyperparameter is the curriculum schedule itself (piece 3), which has no
prior-phase reference to match since it didn't exist before this phase.

Warm start: proj + set_enc weights only (the shared transformer backbone)
are loaded from stage 1's checkpoint, exactly as the paper describes
("pre-train ... and use the learned weights to initialize the transformer,
image and text encoder ... for complementary item retrieval"). embed_ffn
(new, retrieval-only output head) and outfit_token/cp_head (CP-only, unused
from here on) are left at their fresh random initialization.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP, uniformity_loss
import train_lib as tl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

STAGE1_CKPT = MODELS_DIR / "stage1_cp_pretrain.pt"

# Phase 31/32's exact winning configuration -- held constant, not re-tuned.
CONFIG = {
    "input_mode": "image_text",
    "lr": 1.5e-4, "batch_size": 384, "uniformity_weight": 0.1, "margin": 0.2,
    "max_epochs": 100, "patience": 25, "min_delta": 0.0005,
    "num_negatives": 10, "seed": 42,
    "eval_every": 1,
    # Curriculum (piece 3): linear ramp from 0% hard (pure random,
    # same-category) at epoch 0 to 100% hard (mined, visually-similar
    # same-category) at the final epoch -- the brief's own suggested
    # default when the paper's exact schedule isn't reproducible (it names
    # only a coarse two-stage split, no epoch numbers). See
    # implementation_notes.md.
    "curriculum_final_epoch": 99,  # hard_fraction reaches 1.0 here (== max_epochs-1)
}


def hard_fraction_at(epoch, final_epoch):
    return min(1.0, epoch / max(final_epoch, 1))


def main():
    device = tl.get_device()
    print(f"device={device}")
    cat = tl.load_catalog(device)
    base_repr = cat["base_repr"]
    id_to_gidx = cat["id_to_gidx"]
    train_outfits = cat["train_outfits"]
    val_outfits = cat["val_outfits"]
    cat_raw_t = cat["cat_raw_t"]
    cat_row_of = cat["cat_row_of"]

    torch.manual_seed(CONFIG["seed"])
    model = OutfitTransformerSigLIP().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    warm_started = False
    if STAGE1_CKPT.exists():
        sd = torch.load(STAGE1_CKPT, map_location=device)
        own_sd = model.state_dict()
        shared = {k: v for k, v in sd.items() if k.startswith("proj.") or k.startswith("set_enc.")}
        own_sd.update(shared)
        model.load_state_dict(own_sd)
        warm_started = True
        print(f"warm-started proj+set_enc ({len(shared)} tensors) from {STAGE1_CKPT}")
    else:
        print(f"WARNING: {STAGE1_CKPT} not found -- training stage 2 from scratch (no CP warm start).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    steps_per_epoch = max(1, -(-len(train_outfits) // CONFIG["batch_size"]))
    total_steps = CONFIG["max_epochs"] * steps_per_epoch + 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr"], total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )

    state = tl.RetrievalTrainState(cat["train_items_by_cat"], id_to_gidx, cat["neg_cand_train"], seed=CONFIG["seed"])
    val_state = tl.RetrievalTrainState(cat["val_items_by_cat"], id_to_gidx, cat["neg_cand_val"], seed=42)
    # fixed val sample set for the secondary val_loss/D_pos/D_neg diagnostics,
    # evaluated at hard_fraction=1.0 throughout (worst-case/final-regime
    # diagnostic, not used for checkpoint selection -- val Recall@10 is).
    val_samples_fixed = [val_state.make_sample(o, CONFIG["num_negatives"], hard_fraction=1.0) for o in val_outfits]

    best_recall10 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    for epoch in range(CONFIG["max_epochs"]):
        hf = hard_fraction_at(epoch, CONFIG["curriculum_final_epoch"])
        model.train()
        order = list(range(len(train_outfits)))
        state.rng_np.shuffle(order)
        ep_losses, ep_d_pos, ep_d_neg = [], [], []
        for b_start in range(0, len(order), CONFIG["batch_size"]):
            b_idx = order[b_start:b_start + CONFIG["batch_size"]]
            if len(b_idx) < 2:
                continue
            samples = [state.make_sample(train_outfits[i], CONFIG["num_negatives"], hf) for i in b_idx]
            ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, cat_idx = tl.build_retrieval_batch(
                samples, base_repr, cat_row_of, device
            )
            optimizer.zero_grad()
            loss, d_pos, d_neg = tl.compute_retrieval_batch_loss(
                model, cat_raw_t, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, cat_idx,
                CONFIG["margin"], CONFIG["uniformity_weight"], uniformity_loss,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            ep_losses.append(loss.item())
            ep_d_pos.append(d_pos)
            ep_d_neg.append(d_neg)
        train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        model.eval()
        val_losses, val_d_pos, val_d_neg = [], [], []
        with torch.no_grad():
            for b_start in range(0, len(val_samples_fixed), CONFIG["batch_size"]):
                batch = val_samples_fixed[b_start:b_start + CONFIG["batch_size"]]
                if len(batch) < 2:
                    continue
                ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, cat_idx = tl.build_retrieval_batch(
                    batch, base_repr, cat_row_of, device
                )
                loss, d_pos, d_neg = tl.compute_retrieval_batch_loss(
                    model, cat_raw_t, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, cat_idx,
                    CONFIG["margin"], CONFIG["uniformity_weight"], uniformity_loss,
                )
                val_losses.append(loss.item())
                val_d_pos.append(d_pos)
                val_d_neg.append(d_neg)
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        row = {"epoch": epoch, "hard_fraction": hf, "train_loss": train_loss, "val_loss": val_loss,
               "val_D_pos": float(np.mean(val_d_pos)) if val_d_pos else 0.0,
               "val_D_neg": float(np.mean(val_d_neg)) if val_d_neg else 0.0}

        if epoch % CONFIG["eval_every"] == 0 or epoch == CONFIG["max_epochs"] - 1:
            recall, n_total, n_skipped = tl.evaluate_recall_targeted(
                model, base_repr, id_to_gidx, cat_raw_t, cat_row_of, device, bench_key="val",
            )
            row.update({"recall10": recall[10], "recall30": recall[30], "recall50": recall[50]})

            improved = recall[10] > best_recall10 + CONFIG["min_delta"]
            if improved:
                best_recall10 = recall[10]
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

        elapsed = time.time() - t_start
        print(f"epoch {epoch}: hf={hf:.2f} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_D_pos={row['val_D_pos']:.4f} val_D_neg={row['val_D_neg']:.4f} "
              f"recall10={row.get('recall10', float('nan')):.4f} "
              f"elapsed={elapsed:.0f}s best_recall10={best_recall10:.4f}@{best_epoch}")
        curve.append(row)

        if patience_counter >= CONFIG["patience"]:
            print(f"early stop at epoch {epoch} (patience={CONFIG['patience']})")
            break

    wall_time = time.time() - t_start
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "stage2_retrieval_finetune.pt"
    torch.save(best_state, ckpt_path)

    result = {
        "config": CONFIG, "curve": curve, "best_epoch": best_epoch, "best_recall10": best_recall10,
        "n_epochs_run": len(curve), "wall_time_sec": wall_time, "device": device, "n_params": n_params,
        "warm_started_from_stage1": warm_started,
    }
    with open(DATA_DIR / "stage2_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved checkpoint to {ckpt_path}")
    print(f"Saved result to {DATA_DIR / 'stage2_result.json'}")
    print(f"s/epoch (mean): {wall_time / max(len(curve), 1):.1f}")


if __name__ == "__main__":
    main()

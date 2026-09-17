"""
Phase 35, stage 1: compatibility-prediction (CP) pre-training (piece 1).

New stage, no prior-phase reference to hold hyperparameters constant against
(unlike stage 2, which is required to match phase 32's exact winning config
for a clean recipe-only comparison) -- lr/batch/epoch choices here are this
phase's own first-time defaults, documented as such, not tuned.

Runs locally (M4, MPS if available). Checkpoint selection: primary criterion
is val CP accuracy plateau (this stage's own objective); val Recall@10 is
also computed every eval as a secondary diagnostic per the brief -- expected
to be low/uninformative here since embed_ffn (the retrieval output head) is
still at its random initialization and is never trained by the CP objective
at all (CP uses cp_head, a completely separate head). See
stage1_training_log.md for the actual numbers and stage1_training_log.md's
discussion of this expected decoupling.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # see train_lib.py's module docstring note

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP, focal_loss
import train_lib as tl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

CONFIG = {
    "lr": 1.5e-4,          # same optimizer/scheduler SHAPE as stage 2 (AdamW+OneCycleLR); own budget below
    "batch_size": 192,     # 192 outfits/step -> 384 sequences/step (real+fake), matches stage 2's 384 seq/step scale
    "max_epochs": 40,
    "patience": 8,
    "min_delta": 0.0005,   # val CP accuracy improvement threshold
    "seed": 42,
    "eval_every": 1,
}


def main():
    device = tl.get_device()
    print(f"device={device}")
    cat = tl.load_catalog(device)
    base_repr = cat["base_repr"]
    id_to_gidx = cat["id_to_gidx"]
    train_outfits = cat["train_outfits"]
    val_outfits = cat["val_outfits"]

    torch.manual_seed(CONFIG["seed"])
    model = OutfitTransformerSigLIP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params={n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    steps_per_epoch = max(1, -(-len(train_outfits) // CONFIG["batch_size"]))
    total_steps = CONFIG["max_epochs"] * steps_per_epoch + 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr"], total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )

    train_state = tl.CPState(train_outfits, seed=CONFIG["seed"])
    val_state = tl.CPState(val_outfits, seed=42)  # fixed val set, built once, reused every eval
    val_item_lists, val_labels = val_state.make_batch_lists(val_outfits)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        order = list(range(len(train_outfits)))
        train_state.rng_np.shuffle(order)
        ep_losses, ep_accs = [], []
        for b_start in range(0, len(order), CONFIG["batch_size"]):
            b_idx = order[b_start:b_start + CONFIG["batch_size"]]
            if len(b_idx) < 2:
                continue
            batch_outfits = [train_outfits[i] for i in b_idx]
            item_lists, labels = train_state.make_batch_lists(batch_outfits)
            item_vecs, pad_mask, label_t = tl.build_cp_batch(item_lists, labels, base_repr, id_to_gidx, device)

            optimizer.zero_grad()
            loss, acc, _ = tl.compute_cp_loss(model, focal_loss, item_vecs, pad_mask, label_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ep_losses.append(loss.item())
            ep_accs.append(acc)
        train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        train_acc = float(np.mean(ep_accs)) if ep_accs else 0.0

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for b_start in range(0, len(val_item_lists), CONFIG["batch_size"] * 2):
                lst_chunk = val_item_lists[b_start:b_start + CONFIG["batch_size"] * 2]
                lab_chunk = val_labels[b_start:b_start + CONFIG["batch_size"] * 2]
                if len(lst_chunk) < 2:
                    continue
                item_vecs, pad_mask, label_t = tl.build_cp_batch(lst_chunk, lab_chunk, base_repr, id_to_gidx, device)
                loss, acc, _ = tl.compute_cp_loss(model, focal_loss, item_vecs, pad_mask, label_t)
                val_losses.append(loss.item())
                val_accs.append(acc)
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}

        if epoch % CONFIG["eval_every"] == 0 or epoch == CONFIG["max_epochs"] - 1:
            recall, n_total, n_skipped = tl.evaluate_recall_targeted(
                model, base_repr, id_to_gidx, cat["cat_raw_t"], cat["cat_row_of"], device, bench_key="val",
            )
            row.update({"val_recall10": recall[10], "val_recall30": recall[30], "val_recall50": recall[50]})

        improved = val_acc > best_acc + CONFIG["min_delta"]
        if improved:
            best_acc = val_acc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - t_start
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_recall10={row.get('val_recall10', float('nan')):.4f} "
              f"elapsed={elapsed:.0f}s best_acc={best_acc:.4f}@{best_epoch}")
        curve.append(row)

        if patience_counter >= CONFIG["patience"]:
            print(f"early stop at epoch {epoch} (patience={CONFIG['patience']})")
            break

    wall_time = time.time() - t_start
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "stage1_cp_pretrain.pt"
    torch.save(best_state, ckpt_path)

    result = {
        "config": CONFIG, "curve": curve, "best_epoch": best_epoch, "best_val_acc": best_acc,
        "n_epochs_run": len(curve), "wall_time_sec": wall_time, "device": device, "n_params": n_params,
    }
    with open(DATA_DIR / "stage1_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved checkpoint to {ckpt_path}")
    print(f"Saved result to {DATA_DIR / 'stage1_result.json'}")
    print(f"s/epoch (mean): {wall_time / max(len(curve), 1):.1f}")


if __name__ == "__main__":
    main()

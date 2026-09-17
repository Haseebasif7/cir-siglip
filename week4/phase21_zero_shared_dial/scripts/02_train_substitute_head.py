"""
Phase 21, step 3: train the second head, using phase 12c's proven
ranking-distillation objective (KL(teacher || student) over each anchor's
precomputed top-K=50 raw-SigLIP neighbors, `01_build_nn_lookup.py`'s output,
reused unchanged from phase 12c/17 -- it depends only on the raw, frozen
SigLIP embeddings, not on anything any phase trains, so recomputing it would
be identical work for no reason) -- but as the SOLE objective on its own
fully independent parameters, not jointly trained against a complement loss
on a shared trunk the way phase 12c/17/18/18b all were. Since there is no
competing objective touching these parameters, no loss-balancing weight is
needed at all (unlike every prior dial phase in this project, which all
needed a calibrated weight_sub specifically because two losses shared one
trunk) -- weight_sub=1.0 throughout, that is the entire point of testing
zero shared capacity.

Hyperparameters (seed=42, batch=128, lr=1e-3, weight_decay=1e-5,
tau_distill=0.07, K=50, early stopping patience=5/min_delta=1e-4,
max_epochs=100) copied from phase 12c's own substitute-loss settings for
direct comparability, even though there is no complement-loss calibration
step to redo here.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import ProjectionHead, mean_pairwise_cosine

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12C_DIR = BASE_DIR.parent / "phase12c_ranking_distillation"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = PHASE9_DIR / "data" / "positive_edges.json"
NN_LOOKUP_NPZ = PHASE12C_DIR / "data" / "nn_lookup.npz"  # reused unchanged, see docstring
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU_DISTILL = 0.07
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_data():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    nn = np.load(NN_LOOKUP_NPZ, allow_pickle=True)
    nn_item_ids = [str(a) for a in nn["item_ids"]]
    assert nn_item_ids == item_ids, "NN lookup ordering must match embedding ordering"
    nn_indices = nn["indices"]
    nn_sims = nn["sims"].astype(np.float32)

    # anchors: same real Polyvore items used throughout this project (train/val
    # edges' own source items), so the substitute head sees the same item
    # population as the complement head, purely for a fair epoch-vs-epoch
    # comparison -- the loss itself doesn't use the edge's target at all.
    with open(POSITIVE_EDGES_JSON) as f:
        edge_records = json.load(f)
    train_anchors = sorted({e["source"] for e in edge_records if e["split"] == "train"})
    val_anchors = sorted({e["source"] for e in edge_records if e["split"] == "val"})

    return embeddings, nn_indices, nn_sims, idx, item_ids, train_anchors, val_anchors


def substitute_ranking_loss(model, embeddings, idx, anchor_ids, nn_indices, nn_sims):
    anchor_gidx = np.array([idx[a] for a in anchor_ids])
    neighbor_gidx = nn_indices[anchor_gidx]
    B, K = neighbor_gidx.shape

    teacher_sims = torch.tensor(nn_sims[anchor_gidx], device=DEVICE)
    teacher_dist = F.softmax(teacher_sims / TAU_DISTILL, dim=-1)

    anchor_emb = torch.tensor(embeddings[anchor_gidx], device=DEVICE)
    neighbor_emb = torch.tensor(embeddings[neighbor_gidx.reshape(-1)], device=DEVICE)

    z_anchor = model(anchor_emb)
    z_neighbors = model(neighbor_emb).reshape(B, K, -1)

    student_sims = torch.einsum("bd,bkd->bk", z_anchor, z_neighbors)
    student_log_probs = F.log_softmax(student_sims / TAU_DISTILL, dim=-1)

    return F.kl_div(student_log_probs, teacher_dist, reduction="batchmean")


def run_epoch_train(model, optimizer, embeddings, nn_indices, nn_sims, idx, anchors, rng):
    model.train()
    order = list(range(len(anchors)))
    rng.shuffle(order)
    total_loss, n_batches = 0.0, 0

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue
        batch_anchors = [anchors[i] for i in batch_idx]

        loss = substitute_ranking_loss(model, embeddings, idx, batch_anchors, nn_indices, nn_sims)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % 2000 == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, running loss {total_loss/n_batches:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, nn_indices, nn_sims, idx, val_anchors):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(val_anchors), BATCH_SIZE):
        batch_anchors = val_anchors[start:start + BATCH_SIZE]
        if len(batch_anchors) < 2:
            continue
        loss = substitute_ranking_loss(model, embeddings, idx, batch_anchors, nn_indices, nn_sims)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    embeddings, nn_indices, nn_sims, idx, item_ids, train_anchors, val_anchors = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_anchors)} train anchors, "
          f"{len(val_anchors)} val anchors, NN lookup K={nn_indices.shape[1]}. Device: {DEVICE}", flush=True)

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    model = ProjectionHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_loss = run_epoch_train(model, optimizer, embeddings, nn_indices, nn_sims, idx, train_anchors, rng)
        val_loss = run_epoch_val(model, embeddings, nn_indices, nn_sims, idx, val_anchors)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_metric = mean_pairwise_cosine(model(sample_emb))

        curves.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "mean_pairwise_cosine": collapse_metric})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"mean_pairwise_cosine={collapse_metric:.4f}", flush=True)

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best_val_loss={best_val_loss:.4f})", flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "substitute_head.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'substitute_head.pt'} (best_val_loss={best_val_loss:.4f})", flush=True)

    with open(MODELS_DIR / "substitute_head_training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved training curves.", flush=True)


if __name__ == "__main__":
    main()

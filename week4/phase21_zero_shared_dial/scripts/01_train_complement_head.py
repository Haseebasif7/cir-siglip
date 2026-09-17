"""
Phase 21, steps 1-2: train the complement head. This is deliberately an
exact replica of phase 9's `05_train_projection.py`, Model A specifically
(R=8 random negatives, H=0 hard negatives, seed=42, identical data,
identical hyperparameters, identical training-loop code path) -- the only
difference from literally re-running phase 9's own script is the output
checkpoint's name and location (this phase's own `models/` folder, not
phase 9's), so this phase's own folder is self-contained per this
project's standing convention. If this produces a result "statistically
indistinguishable" from phase 9's own checkpoint (verified directly in
`03_verify_complement_endpoint.py`, not assumed), that confirms there is
truly no other objective touching these parameters -- the entire premise
this phase's zero-shared-capacity design rests on.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import ProjectionHead, mnrl_loss, mean_pairwise_cosine

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = PHASE9_DIR / "data" / "positive_edges.json"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU = 0.07
R_NEG = 8  # random negatives only, matching phase 9's Model A exactly
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

    with open(POSITIVE_EDGES_JSON) as f:
        edge_records = json.load(f)
    train_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "train"]
    val_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "val"]

    all_positive_targets = {}
    for src, tgt in train_edges + val_edges:
        all_positive_targets.setdefault(src, set()).add(tgt)

    return embeddings, idx, item_ids, train_edges, val_edges, all_positive_targets


def sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_item, k):
    negs = []
    exclude = positive_set | {anchor}
    tries, max_tries = 0, k * 20
    while len(negs) < k and tries < max_tries:
        cand = idx_to_item[rng.randrange(n_items)]
        if cand not in exclude and cand not in negs:
            negs.append(cand)
        tries += 1
    while len(negs) < k:
        cand = idx_to_item[rng.randrange(n_items)]
        negs.append(cand)
    return negs


def build_negatives_for_edges(edges, all_positive_targets, item_ids, rng, k):
    n_items = len(item_ids)
    out = []
    for anchor, _ in edges:
        positive_set = all_positive_targets.get(anchor, set())
        out.append(sample_random_negatives(rng, anchor, positive_set, n_items, item_ids, k))
    return out


def build_inbatch_mask(batch_edges, all_positive_targets):
    B = len(batch_edges)
    mask = np.zeros((B, B), dtype=bool)
    for i, (anchor_i, _) in enumerate(batch_edges):
        positives_i = all_positive_targets.get(anchor_i, set())
        for j, (_, positive_j) in enumerate(batch_edges):
            if i != j and positive_j in positives_i:
                mask[i, j] = True
    return mask


def run_epoch_train(model, optimizer, embeddings, idx, edges, all_positive_targets, item_ids, rng):
    model.train()
    order = list(range(len(edges)))
    rng.shuffle(order)
    total_loss, n_batches = 0.0, 0

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue
        batch_edges = [edges[i] for i in batch_idx]
        anchors = [e[0] for e in batch_edges]
        positives = [e[1] for e in batch_edges]
        neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

        a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
        p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
        extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
        extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

        z_a = model(a_emb)
        z_p = model(p_emb)
        B, K, D = extra_emb.shape
        z_extra = model(extra_emb.reshape(B * K, D)).reshape(B, K, -1)

        mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)

        loss = mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % 2000 == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, running loss {total_loss/n_batches:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(val_edges), BATCH_SIZE):
        batch_edges = val_edges[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        anchors = [e[0] for e in batch_edges]
        positives = [e[1] for e in batch_edges]
        neg_lists = frozen_val_negs[start:start + BATCH_SIZE]

        a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
        p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
        extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
        extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

        z_a = model(a_emb)
        z_p = model(p_emb)
        B, K, D = extra_emb.shape
        z_extra = model(extra_emb.reshape(B * K, D)).reshape(B, K, -1)
        mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)

        loss = mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    embeddings, idx, item_ids, train_edges, val_edges, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, "
          f"{len(val_edges)} val edges. Device: {DEVICE}", flush=True)

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)

    model = ProjectionHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    frozen_val_negs = build_negatives_for_edges(val_edges, all_positive_targets, item_ids, val_rng, R_NEG)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_loss = run_epoch_train(model, optimizer, embeddings, idx, train_edges, all_positive_targets, item_ids, rng)
        val_loss = run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets)

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
    torch.save(model.state_dict(), MODELS_DIR / "complement_head.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'complement_head.pt'} (best_val_loss={best_val_loss:.4f})", flush=True)

    with open(MODELS_DIR / "complement_head_training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved training curves.", flush=True)


if __name__ == "__main__":
    main()

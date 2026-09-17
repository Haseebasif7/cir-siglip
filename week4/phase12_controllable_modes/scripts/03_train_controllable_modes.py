"""
Phase 12, step 3: train the controllable mode-embedding mechanism.

One ControllableProjectionHead (see model.py), two mode vectors, trained
JOINTLY in a single loop -- every training step applies both mode-specific
loss terms to the same shared base projection:

  - Complement mode (alpha=0): identical objective to phase 9's Model A --
    MNRL / InfoNCE against real Polyvore co-outfit positive pairs
    (data/positive_edges.json, reused directly from phase 9, no rebuilding),
    random negatives only (R=8, H=0, same as phase 9 Model A), same
    hyperparameters (batch 128, lr 1e-3, wd 1e-5, tau 0.07).
  - Substitute mode (alpha=1): a distillation objective, NOT a contrastive
    one -- the mode-conditioned pairwise similarity structure of the current
    training batch's anchor items is regressed (MSE) toward their pairwise
    raw-SigLIP cosine similarity. This directly encodes "should behave like
    plain visual similarity" as its own loss term rather than reusing MNRL
    with a different target, since substitute mode has no notion of
    "negative" -- everything is scored by how similar it already looks.

Both losses are summed unweighted (lambda=1.0, no tuning attempted -- this is
a cheap first test per the brief, not a hyperparameter search) and backpropagate
through the single shared `net`; only the two mode vectors receive gradient
exclusively from their own loss term.

Early stopping/best-checkpoint selection uses complement-mode validation loss
only (matching phase 9's convention for its MNRL-trained models) -- substitute
validation loss is still logged every epoch as a diagnostic, but is a
regression loss on a different scale (MSE, not cross-entropy) so is not
combined into a single stopping criterion.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ControllableProjectionHead, mean_pairwise_cosine, mnrl_loss

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
R_NEG = 8  # random negatives only, matching phase 9 Model A
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


def substitute_distill_loss(model, anchor_emb_raw_t, anchor_emb_raw_np):
    """Regress mode=1.0 (substitute) pairwise cosine similarity of the batch's
    anchor items toward their pairwise raw-SigLIP cosine similarity."""
    z_sub = model(anchor_emb_raw_t, alpha=1.0)  # (B, D), L2-normalized
    sim_sub = z_sub @ z_sub.T
    sim_raw = torch.tensor(anchor_emb_raw_np @ anchor_emb_raw_np.T, device=DEVICE)
    return F.mse_loss(sim_sub, sim_raw)


def run_epoch_train(model, optimizer, embeddings, idx, edges, all_positive_targets, item_ids, rng):
    model.train()
    order = list(range(len(edges)))
    rng.shuffle(order)
    total_comp, total_sub, n_batches = 0.0, 0.0, 0

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue
        batch_edges = [edges[i] for i in batch_idx]
        anchors = [e[0] for e in batch_edges]
        positives = [e[1] for e in batch_edges]
        neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R_NEG)

        a_emb_np = embeddings[[idx[a] for a in anchors]]
        a_emb = torch.tensor(a_emb_np, device=DEVICE)
        p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
        extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
        extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

        # complement mode (alpha=0): MNRL against real co-outfit positives
        z_a = model(a_emb, alpha=0.0)
        z_p = model(p_emb, alpha=0.0)
        B, K, D = extra_emb.shape
        z_extra = model(extra_emb.reshape(B * K, D), alpha=0.0).reshape(B, K, -1)
        mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
        complement_loss = mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU)

        # substitute mode (alpha=1): distill raw SigLIP pairwise similarity
        substitute_loss = substitute_distill_loss(model, a_emb, a_emb_np)

        loss = complement_loss + substitute_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_comp += complement_loss.item()
        total_sub += substitute_loss.item()
        n_batches += 1
        if n_batches % 2000 == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, "
                  f"complement={total_comp/n_batches:.4f} substitute={total_sub/n_batches:.4f}")

    return total_comp / max(n_batches, 1), total_sub / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets):
    model.eval()
    total_comp, total_sub, n_batches = 0.0, 0.0, 0
    for start in range(0, len(val_edges), BATCH_SIZE):
        batch_edges = val_edges[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        anchors = [e[0] for e in batch_edges]
        positives = [e[1] for e in batch_edges]
        neg_lists = frozen_val_negs[start:start + BATCH_SIZE]

        a_emb_np = embeddings[[idx[a] for a in anchors]]
        a_emb = torch.tensor(a_emb_np, device=DEVICE)
        p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
        extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
        extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

        z_a = model(a_emb, alpha=0.0)
        z_p = model(p_emb, alpha=0.0)
        B, K, D = extra_emb.shape
        z_extra = model(extra_emb.reshape(B * K, D), alpha=0.0).reshape(B, K, -1)
        mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
        complement_loss = mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU)
        substitute_loss = substitute_distill_loss(model, a_emb, a_emb_np)

        total_comp += complement_loss.item()
        total_sub += substitute_loss.item()
        n_batches += 1

    return total_comp / max(n_batches, 1), total_sub / max(n_batches, 1)


def main():
    embeddings, idx, item_ids, train_edges, val_edges, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, "
          f"{len(val_edges)} val edges. Device: {DEVICE}")

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)

    model = ControllableProjectionHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    frozen_val_negs = build_negatives_for_edges(val_edges, all_positive_targets, item_ids, val_rng, R_NEG)

    best_val_comp = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_comp, train_sub = run_epoch_train(model, optimizer, embeddings, idx, train_edges,
                                                  all_positive_targets, item_ids, rng)
        val_comp, val_sub = run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_comp = mean_pairwise_cosine(model(sample_emb, alpha=0.0))
            collapse_sub = mean_pairwise_cosine(model(sample_emb, alpha=1.0))

        curves.append({
            "epoch": epoch, "train_complement_loss": train_comp, "train_substitute_loss": train_sub,
            "val_complement_loss": val_comp, "val_substitute_loss": val_sub,
            "mean_pairwise_cosine_complement": collapse_comp, "mean_pairwise_cosine_substitute": collapse_sub,
        })
        print(f"  epoch {epoch}: train_comp={train_comp:.4f} train_sub={train_sub:.4f} "
              f"val_comp={val_comp:.4f} val_sub={val_sub:.4f} "
              f"collapse_comp={collapse_comp:.4f} collapse_sub={collapse_sub:.4f}")

        if val_comp < best_val_comp - MIN_DELTA:
            best_val_comp = val_comp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best_val_comp={best_val_comp:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "controllable_modes.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / 'controllable_modes.pt'} (best_val_comp={best_val_comp:.4f})")

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

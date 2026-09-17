"""
Phase 10, step 3: train the color-invariant compatibility model.

Single model only (per the brief: reuse phase 9 Model A's exact setup --
random negatives only, R=8/H=0 -- do not revisit the hard-negative question
here, that stays phase 9's separate thread). Same architecture, same MNRL
compatibility loss, same hyperparameters, same positive edges (reused
directly from phase 9's data/positive_edges.json) as phase 9's Model A.

The one new piece: a second loss term. For every batch, in addition to the
compatibility loss (over anchors/positives/negatives), every unique item
appearing in the batch as an anchor or positive is also scored for
invariance -- its projection from the phase 9 original SigLIP embedding vs.
its projection from this phase's color-perturbed-twin SigLIP embedding
should point the same direction. Combined with equal fixed weighting
(total_loss = compat_loss + invariance_loss), per the brief's instruction
to test one well-motivated configuration rather than sweep weightings.

Negatives are drawn from the ORIGINAL (non-perturbed) embedding space only
-- the invariance signal only needs anchor/positive pairs, matching the
brief's "every training image, paired with its own color-perturbed twin"
framing; negatives don't need a perturbed counterpart for this loss.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import ProjectionHead, mnrl_loss, invariance_loss, mean_pairwise_cosine

PHASE9_DIR = Path(__file__).resolve().parent.parent.parent / "phase9_polyvore_compatibility"
ORIG_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = PHASE9_DIR / "data" / "positive_edges.json"

BASE_DIR = Path(__file__).resolve().parent.parent
PERTURBED_EMBEDDINGS_NPZ = BASE_DIR / "embeddings" / "siglip_perturbed.npz"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
TAU = 0.07
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-4
R_NEG = 8  # random negatives only, matching phase 9 Model A
INVARIANCE_WEIGHT = 1.0  # equal weighting with compat loss, per the brief
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_data():
    orig = np.load(ORIG_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in orig["item_ids"]]
    embeddings = orig["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    pert = np.load(PERTURBED_EMBEDDINGS_NPZ, allow_pickle=True)
    pert_item_ids = [str(a) for a in pert["item_ids"]]
    pert_embeddings = pert["embeddings"]
    pert_norms = np.linalg.norm(pert_embeddings, axis=1, keepdims=True)
    pert_embeddings = (pert_embeddings / pert_norms).astype(np.float32)
    pert_idx = {a: i for i, a in enumerate(pert_item_ids)}

    with open(POSITIVE_EDGES_JSON) as f:
        edge_records = json.load(f)
    train_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "train"]
    val_edges = [(e["source"], e["target"]) for e in edge_records if e["split"] == "val"]

    all_positive_targets = {}
    for src, tgt in train_edges + val_edges:
        all_positive_targets.setdefault(src, set()).add(tgt)

    n_missing_pert = sum(1 for a, b in train_edges + val_edges if a not in pert_idx or b not in pert_idx)
    print(f"{n_missing_pert} edge endpoints missing a perturbed twin (expected 0).")

    return embeddings, idx, item_ids, pert_embeddings, pert_idx, train_edges, val_edges, all_positive_targets


def sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_item, k):
    negs = []
    exclude = positive_set | {anchor}
    tries = 0
    max_tries = k * 20
    while len(negs) < k and tries < max_tries:
        cand = idx_to_item[rng.randrange(n_items)]
        if cand not in exclude and cand not in negs:
            negs.append(cand)
        tries += 1
    while len(negs) < k:
        cand = idx_to_item[rng.randrange(n_items)]
        negs.append(cand)
    return negs


def build_negatives_for_edges(edges, all_positive_targets, item_ids, rng, R):
    n_items = len(item_ids)
    out = []
    for anchor, _ in edges:
        positive_set = all_positive_targets.get(anchor, set())
        rand_negs = sample_random_negatives(rng, anchor, positive_set, n_items, item_ids, R)
        out.append(rand_negs)
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


def compute_batch_losses(model, batch_edges, neg_lists, embeddings, idx, pert_embeddings, pert_idx,
                          all_positive_targets):
    anchors = [e[0] for e in batch_edges]
    positives = [e[1] for e in batch_edges]

    a_emb = torch.tensor(embeddings[[idx[a] for a in anchors]], device=DEVICE)
    p_emb = torch.tensor(embeddings[[idx[p] for p in positives]], device=DEVICE)
    extra_idx = [[idx[n] for n in negs] for negs in neg_lists]
    extra_emb = torch.tensor(embeddings[np.array(extra_idx)], device=DEVICE)

    z_a = model(a_emb)
    z_p = model(p_emb)
    B, K, D = extra_emb.shape
    z_extra = model(extra_emb.reshape(B * K, D)).reshape(B, K, -1)

    mask = torch.tensor(build_inbatch_mask(batch_edges, all_positive_targets), device=DEVICE)
    compat_loss = mnrl_loss(z_a, z_p, z_extra, mask, tau=TAU)

    # invariance: every unique item in this batch as anchor or positive
    unique_items = sorted(set(anchors) | set(positives))
    orig_idx_u = [idx[i] for i in unique_items]
    pert_idx_u = [pert_idx[i] for i in unique_items]
    u_orig_emb = torch.tensor(embeddings[orig_idx_u], device=DEVICE)
    u_pert_emb = torch.tensor(pert_embeddings[pert_idx_u], device=DEVICE)
    z_u_orig = model(u_orig_emb)
    z_u_pert = model(u_pert_emb)
    inv_loss = invariance_loss(z_u_orig, z_u_pert)

    total_loss = compat_loss + INVARIANCE_WEIGHT * inv_loss
    return total_loss, compat_loss, inv_loss, z_a


def run_epoch_train(model, optimizer, embeddings, idx, edges, all_positive_targets, item_ids,
                     pert_embeddings, pert_idx, rng, R):
    model.train()
    order = list(range(len(edges)))
    rng.shuffle(order)
    total_loss, total_compat, total_inv, n_batches = 0.0, 0.0, 0.0, 0

    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue
        batch_edges = [edges[i] for i in batch_idx]
        neg_lists = build_negatives_for_edges(batch_edges, all_positive_targets, item_ids, rng, R)

        loss, compat_loss, inv_loss, _ = compute_batch_losses(
            model, batch_edges, neg_lists, embeddings, idx, pert_embeddings, pert_idx, all_positive_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_compat += compat_loss.item()
        total_inv += inv_loss.item()
        n_batches += 1
        if n_batches % 2000 == 0:
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, "
                  f"running total={total_loss/n_batches:.4f} compat={total_compat/n_batches:.4f} "
                  f"inv={total_inv/n_batches:.4f}")

    return total_loss / max(n_batches, 1), total_compat / max(n_batches, 1), total_inv / max(n_batches, 1)


@torch.no_grad()
def run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets, item_ids,
                   pert_embeddings, pert_idx):
    model.eval()
    total_loss, total_compat, total_inv, n_batches = 0.0, 0.0, 0.0, 0
    for start in range(0, len(val_edges), BATCH_SIZE):
        batch_edges = val_edges[start:start + BATCH_SIZE]
        if len(batch_edges) < 2:
            continue
        neg_lists = frozen_val_negs[start:start + BATCH_SIZE]

        loss, compat_loss, inv_loss, _ = compute_batch_losses(
            model, batch_edges, neg_lists, embeddings, idx, pert_embeddings, pert_idx, all_positive_targets)

        total_loss += loss.item()
        total_compat += compat_loss.item()
        total_inv += inv_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_compat / max(n_batches, 1), total_inv / max(n_batches, 1)


def main():
    (embeddings, idx, item_ids, pert_embeddings, pert_idx, train_edges, val_edges,
     all_positive_targets) = load_data()
    print(f"Loaded {len(item_ids)} original embeddings, {len(pert_embeddings)} perturbed embeddings, "
          f"{len(train_edges)} train edges, {len(val_edges)} val edges.")
    print(f"Using device: {DEVICE}")

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
        train_loss, train_compat, train_inv = run_epoch_train(
            model, optimizer, embeddings, idx, train_edges, all_positive_targets, item_ids,
            pert_embeddings, pert_idx, rng, R_NEG)
        val_loss, val_compat, val_inv = run_epoch_val(
            model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets, item_ids,
            pert_embeddings, pert_idx)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_metric = mean_pairwise_cosine(model(sample_emb))

        curves.append({"epoch": epoch, "train_loss": train_loss, "train_compat_loss": train_compat,
                        "train_invariance_loss": train_inv, "val_loss": val_loss,
                        "val_compat_loss": val_compat, "val_invariance_loss": val_inv,
                        "mean_pairwise_cosine": collapse_metric})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} (compat={train_compat:.4f} inv={train_inv:.4f}) "
              f"val_loss={val_loss:.4f} (compat={val_compat:.4f} inv={val_inv:.4f}) "
              f"mean_pairwise_cosine={collapse_metric:.4f}")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (patience={PATIENCE} exceeded, "
                      f"best_val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    out_path = MODELS_DIR / "model_color_invariant.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved best checkpoint to {out_path} (best_val_loss={best_val_loss:.4f})")

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump({"model_color_invariant": curves}, f, indent=2)
    print(f"Saved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

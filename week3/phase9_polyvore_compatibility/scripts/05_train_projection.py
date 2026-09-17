"""
Phase 9, step 4: train the same architecture as phase 8 (single-variable
test -- only the training data source changes, from Amazon also_buy edges
to Polyvore co-outfit pairs). Identical hyperparameters to phase 7/8:
ProjectionHead 768->256->128, MNRL loss with in-batch false-negative
masking, Adam lr=1e-3/wd=1e-5, batch 128, tau=0.07, early stopping
patience=5/min_delta=1e-4.

Two variants:
  Model A: random negatives only (R=8, H=0)
  Model B: random + explicit hard negatives (R=4, H=4)
Same caveat as phases 7/8 applies: an item never co-outfitted with another
doesn't necessarily mean incompatible, just that combination was never
assembled. Testing hard negatives here regardless, per the brief -- don't
assume they'll work just because the data is cleaner overall.

At ~1.37M train edges (vs phase 8's ~15k), this is a much larger run --
benchmarked at ~35ms/batch on this hardware, ~6-7 min/epoch for the full
train set.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import ProjectionHead, mnrl_loss, mean_pairwise_cosine

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_NPZ = BASE_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = BASE_DIR / "data" / "positive_edges.json"
HARD_NEG_JSON = BASE_DIR / "data" / "hard_negative_candidates.json"
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

    with open(HARD_NEG_JSON) as f:
        hard_neg_raw = json.load(f)
    hard_neg = {a: [c[0] for c in cands] for a, cands in hard_neg_raw.items()}

    all_positive_targets = {}
    for src, tgt in train_edges + val_edges:
        all_positive_targets.setdefault(src, set()).add(tgt)

    return embeddings, idx, item_ids, train_edges, val_edges, hard_neg, all_positive_targets


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


def build_negatives_for_edges(edges, hard_neg, all_positive_targets, item_ids, rng, R, H):
    idx_to_item = item_ids
    n_items = len(item_ids)
    out = []
    for anchor, _ in edges:
        positive_set = all_positive_targets.get(anchor, set())
        rand_negs = sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_item, R)
        if H > 0:
            candidates = hard_neg.get(anchor, [])
            if len(candidates) >= H:
                hard_negs = rng.sample(candidates, H)
            else:
                hard_negs = list(candidates)
                extra_needed = H - len(hard_negs)
                hard_negs += sample_random_negatives(
                    rng, anchor, positive_set | set(hard_negs), n_items, idx_to_item, extra_needed
                )
        else:
            hard_negs = []
        out.append(rand_negs + hard_negs)
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


def run_epoch_train(model, optimizer, embeddings, idx, edges, hard_neg, all_positive_targets,
                     item_ids, rng, R, H):
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
        neg_lists = build_negatives_for_edges(batch_edges, hard_neg, all_positive_targets, item_ids, rng, R, H)

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
            print(f"    ...batch {n_batches}/{len(order)//BATCH_SIZE}, running loss {total_loss/n_batches:.4f}")

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


def train_model(name, R, H, embeddings, idx, item_ids, train_edges, val_edges, hard_neg, all_positive_targets):
    print(f"\n=== Training {name} (R={R} random negs, H={H} hard negs) ===")
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)

    model = ProjectionHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    frozen_val_negs = build_negatives_for_edges(val_edges, hard_neg, all_positive_targets, item_ids, val_rng, R, H)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_loss = run_epoch_train(model, optimizer, embeddings, idx, train_edges, hard_neg,
                                      all_positive_targets, item_ids, rng, R, H)
        val_loss = run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(item_ids), size=min(256, len(item_ids)), replace=False)
            sample_emb = torch.tensor(embeddings[sample_idx], device=DEVICE)
            collapse_metric = mean_pairwise_cosine(model(sample_emb))

        curves.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "mean_pairwise_cosine": collapse_metric})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
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
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    print(f"Saved best checkpoint to {MODELS_DIR / f'{name}.pt'} (best_val_loss={best_val_loss:.4f})")
    return curves


def main():
    embeddings, idx, item_ids, train_edges, val_edges, hard_neg, all_positive_targets = load_data()
    print(f"Loaded {len(item_ids)} embeddings, {len(train_edges)} train edges, {len(val_edges)} val edges.")
    print(f"Using device: {DEVICE}")

    curves_a = train_model("model_a_random_negs", R=8, H=0, embeddings=embeddings, idx=idx,
                            item_ids=item_ids, train_edges=train_edges, val_edges=val_edges,
                            hard_neg=hard_neg, all_positive_targets=all_positive_targets)
    curves_b = train_model("model_b_hard_negs", R=4, H=4, embeddings=embeddings, idx=idx,
                            item_ids=item_ids, train_edges=train_edges, val_edges=val_edges,
                            hard_neg=hard_neg, all_positive_targets=all_positive_targets)

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump({"model_a_random_negs": curves_a, "model_b_hard_negs": curves_b}, f, indent=2)
    print(f"\nSaved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

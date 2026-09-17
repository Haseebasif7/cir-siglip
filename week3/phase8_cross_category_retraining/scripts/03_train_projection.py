"""
Phase 8, step 3: retrain the same architecture as phase 7 (small MLP
projection head on frozen SigLIP, 768->256->128, L2-normalized), same
training setup (MNRL loss, in-batch false-negative masking, early stopping),
but with the POSITIVE edges restricted to heterogeneous dyads (step 2's
output) instead of phase 7's full also_buy edge set.

Important asymmetry, deliberate: the positive TRAINING edges are
heterogeneous-only, but the exclusion set used for in-batch false-negative
masking and hard/random-negative sampling is phase 7's FULL also_buy edge
set (all types, train+val combined) -- not just the heterogeneous subset.
A same-type also_buy partner is still a real relationship even though this
phase isn't training on it as a positive; treating it as a valid negative
target would be factually wrong and would reintroduce the false-negative
contamination this phase exists to avoid (see model.py / step 2 docstrings).

Trains the same two ablation variants:
  Model A: random negatives only (R=8, H=0)
  Model B: random + explicit hard negatives (R=4, H=4)
Worth retesting hard negatives even though they failed in phase 7, since the
contamination that likely caused that failure (near-duplicate substitute
pairs polluting the positive set) should be substantially reduced now.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import ProjectionHead, mnrl_loss, mean_pairwise_cosine

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent / "phase7_learned_compatibility"

EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"  # reused as-is, same cleaned pool
PHASE7_ALL_POSITIVE_EDGES = PHASE7_DIR / "data" / "positive_edges.json"  # for exclusion set only
HETERO_POSITIVE_EDGES_JSON = BASE_DIR / "data" / "heterogeneous_positive_edges.json"  # actual training positives
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
    asins_arr = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(asins_arr)}

    with open(HETERO_POSITIVE_EDGES_JSON) as f:
        hetero_records = json.load(f)
    train_edges = [(e["source"], e["target"]) for e in hetero_records if e["split"] == "train"]
    val_edges = [(e["source"], e["target"]) for e in hetero_records if e["split"] == "val"]

    with open(HARD_NEG_JSON) as f:
        hard_neg_raw = json.load(f)
    hard_neg = {a: [c[0] for c in cands] for a, cands in hard_neg_raw.items()}

    # exclusion set for masking/negative-sampling: ALL also_buy positives
    # (any type), not just the heterogeneous ones being trained on -- see
    # module docstring for why this must be the full set.
    with open(PHASE7_ALL_POSITIVE_EDGES) as f:
        phase7_all_edges = json.load(f)
    all_positive_targets = {}
    for e in phase7_all_edges:
        all_positive_targets.setdefault(e["source"], set()).add(e["target"])

    return embeddings, idx, asins_arr, train_edges, val_edges, hard_neg, all_positive_targets


def sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_asin, k):
    negs = []
    exclude = positive_set | {anchor}
    tries = 0
    max_tries = k * 20
    while len(negs) < k and tries < max_tries:
        cand = idx_to_asin[rng.randrange(n_items)]
        if cand not in exclude and cand not in negs:
            negs.append(cand)
        tries += 1
    while len(negs) < k:
        cand = idx_to_asin[rng.randrange(n_items)]
        negs.append(cand)
    return negs


def build_negatives_for_edges(edges, hard_neg, all_positive_targets, asins_arr, rng, R, H):
    idx_to_asin = asins_arr
    n_items = len(asins_arr)
    out = []
    for anchor, _ in edges:
        positive_set = all_positive_targets.get(anchor, set())
        rand_negs = sample_random_negatives(rng, anchor, positive_set, n_items, idx_to_asin, R)
        if H > 0:
            candidates = hard_neg.get(anchor, [])
            if len(candidates) >= H:
                hard_negs = rng.sample(candidates, H)
            else:
                hard_negs = list(candidates)
                extra_needed = H - len(hard_negs)
                hard_negs += sample_random_negatives(
                    rng, anchor, positive_set | set(hard_negs), n_items, idx_to_asin, extra_needed
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
                     asins_arr, rng, R, H):
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
        neg_lists = build_negatives_for_edges(batch_edges, hard_neg, all_positive_targets, asins_arr, rng, R, H)

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


def train_model(name, R, H, embeddings, idx, asins_arr, train_edges, val_edges, hard_neg, all_positive_targets):
    print(f"\n=== Training {name} (R={R} random negs, H={H} hard negs) ===")
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1000)

    model = ProjectionHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    frozen_val_negs = build_negatives_for_edges(val_edges, hard_neg, all_positive_targets, asins_arr, val_rng, R, H)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        train_loss = run_epoch_train(model, optimizer, embeddings, idx, train_edges, hard_neg,
                                      all_positive_targets, asins_arr, rng, R, H)
        val_loss = run_epoch_val(model, embeddings, idx, val_edges, frozen_val_negs, all_positive_targets)

        with torch.no_grad():
            sample_idx = np.random.default_rng(epoch).choice(len(asins_arr), size=min(256, len(asins_arr)), replace=False)
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
    embeddings, idx, asins_arr, train_edges, val_edges, hard_neg, all_positive_targets = load_data()
    print(f"Loaded {len(asins_arr)} embeddings, {len(train_edges)} heterogeneous train edges, "
          f"{len(val_edges)} heterogeneous val edges.")
    print(f"Using device: {DEVICE}")

    curves_a = train_model("model_a_random_negs", R=8, H=0, embeddings=embeddings, idx=idx,
                            asins_arr=asins_arr, train_edges=train_edges, val_edges=val_edges,
                            hard_neg=hard_neg, all_positive_targets=all_positive_targets)
    curves_b = train_model("model_b_hard_negs", R=4, H=4, embeddings=embeddings, idx=idx,
                            asins_arr=asins_arr, train_edges=train_edges, val_edges=val_edges,
                            hard_neg=hard_neg, all_positive_targets=all_positive_targets)

    with open(MODELS_DIR / "training_curves.json", "w") as f:
        json.dump({"model_a_random_negs": curves_a, "model_b_hard_negs": curves_b}, f, indent=2)
    print(f"\nSaved {MODELS_DIR / 'training_curves.json'}")


if __name__ == "__main__":
    main()

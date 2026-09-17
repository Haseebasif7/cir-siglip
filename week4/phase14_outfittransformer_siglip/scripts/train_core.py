"""
Phase 14: training loop for OutfitTransformer's set-encoder mechanism on a
frozen SigLIP backbone. Reuses phase 13's outfit lists (train/val split,
just item-id lists per outfit -- format confirmed reusable directly, see
../architecture_notes.md) and phase 9's precomputed SigLIP embeddings.

No negative mining is needed here (unlike CSA-Net's phases 13/13b/13c) --
the repo's own training objective (InBatchTripletMarginLoss) uses OTHER
outfits' target items already present in the same batch as negatives, so
phase 13's negative_candidates.json is intentionally not used in this phase.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from model import OutfitTransformerSigLIP, in_batch_triplet_loss, uniformity_loss

UNIFORMITY_WEIGHT = 1.0  # only applied if enabled -- see run_training's use_uniformity flag


class TrainState:
    def __init__(self, embeddings_npz, training_data_path, seed=0):
        self.rng = random.Random(seed)

        data = np.load(embeddings_npz, allow_pickle=True)
        item_ids = [str(a) for a in data["item_ids"]]
        embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        self.idx = {a: i for i, a in enumerate(item_ids)}
        self.embeddings = embeddings  # (N, 768), L2-normalized

        with open(training_data_path) as f:
            td = json.load(f)
        self.categories = td["categories"]
        self.train_outfits = td["train_outfits"]
        self.val_outfits = td["val_outfits"]

    def make_sample(self, outfit_record):
        """Random target item + rest-of-outfit context -- exactly the
        repo's PolyvoreTripletDataset.__getitem__."""
        items = outfit_record["items"]
        target = self.rng.choice(items)
        context = [i for i in items if i != target]
        return {"context_items": context, "target": target}


def build_batch(state, samples, device):
    """Builds padded context-token tensors and target-item tensors for a
    batch of samples. Returns (ctx_vecs, ctx_mask, target_vecs) where
    ctx_vecs is (B, Lmax, 768) raw SigLIP vectors (zero-padded), ctx_mask is
    (B, Lmax) bool (True=pad), target_vecs is (B, 768)."""
    B = len(samples)
    lengths = [len(s["context_items"]) for s in samples]
    Lmax = max(lengths)

    ctx_vecs = np.zeros((B, Lmax, state.embeddings.shape[1]), dtype=np.float32)
    ctx_mask = np.ones((B, Lmax), dtype=bool)
    for i, s in enumerate(samples):
        for j, item in enumerate(s["context_items"]):
            ctx_vecs[i, j] = state.embeddings[state.idx[item]]
            ctx_mask[i, j] = False

    target_vecs = np.stack([state.embeddings[state.idx[s["target"]]] for s in samples])

    return (
        torch.tensor(ctx_vecs, device=device),
        torch.tensor(ctx_mask, device=device),
        torch.tensor(target_vecs, device=device),
    )


def compute_batch_loss(model, ctx_vecs, ctx_mask, target_vecs, use_uniformity):
    B, Lmax, D = ctx_vecs.shape
    ctx_tokens = model.encode_item_tokens(ctx_vecs.view(B * Lmax, D)).view(B, Lmax, -1)
    query_emb = model.embed_query(ctx_tokens, ctx_mask)

    target_tokens = model.encode_item_tokens(target_vecs)
    answer_emb = model.embed_item_alone(target_tokens)

    triplet_loss, d_pos, d_neg = in_batch_triplet_loss(query_emb, answer_emb)
    total_loss = triplet_loss
    if use_uniformity:
        total_loss = total_loss + UNIFORMITY_WEIGHT * uniformity_loss(answer_emb)
    return total_loss, d_pos, d_neg


def run_training(embeddings_npz, training_data_path, out_dir, device,
                  max_epochs, batch_size, lr, patience,
                  n_train_outfits=None, n_val_outfits=None, log_every=50,
                  use_uniformity=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState(embeddings_npz, training_data_path, seed=0)
    train_outfits = [o for o in state.train_outfits if len(o["items"]) >= 2]
    val_outfits = [o for o in state.val_outfits if len(o["items"]) >= 2]
    if n_train_outfits:
        train_outfits = train_outfits[:n_train_outfits]
    if n_val_outfits:
        val_outfits = val_outfits[:n_val_outfits]

    model = OutfitTransformerSigLIP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # ceil division to match the actual number of batches the loop below produces
    # (range(0, N, batch_size) yields ceil(N/batch_size) chunks) -- a floor-division
    # mismatch here undercounts steps and makes OneCycleLR raise once it's stepped
    # past its declared total_steps, discovered by this phase's own smoke test
    steps_per_epoch = max(1, math.ceil(len(train_outfits) / batch_size))
    total_steps = max_epochs * steps_per_epoch + 2  # small safety buffer
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )

    val_state = TrainState(embeddings_npz, training_data_path, seed=42)
    val_samples_fixed = [val_state.make_sample(o) for o in val_outfits]

    history = []
    best_val_loss = float("inf")
    epochs_since_improve = 0
    step = 0

    for epoch in range(max_epochs):
        model.train()
        order = list(range(len(train_outfits)))
        state.rng.shuffle(order)
        epoch_losses = []
        for b_start in range(0, len(order), batch_size):
            b_idx = order[b_start:b_start + batch_size]
            if len(b_idx) < 2:
                continue  # in-batch negatives need at least 2 outfits
            samples = [state.make_sample(train_outfits[i]) for i in b_idx]
            ctx_vecs, ctx_mask, target_vecs = build_batch(state, samples, device)

            optimizer.zero_grad()
            loss, d_pos, d_neg = compute_batch_loss(model, ctx_vecs, ctx_mask, target_vecs, use_uniformity)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f} "
                      f"D_pos={d_pos:.4f} D_neg={d_neg:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)

        model.eval()
        val_losses, val_d_pos, val_d_neg = [], [], []
        with torch.no_grad():
            for b_start in range(0, len(val_samples_fixed), batch_size):
                batch = val_samples_fixed[b_start:b_start + batch_size]
                if len(batch) < 2:
                    continue
                ctx_vecs, ctx_mask, target_vecs = build_batch(val_state, batch, device)
                loss, d_pos, d_neg = compute_batch_loss(model, ctx_vecs, ctx_mask, target_vecs, use_uniformity)
                val_losses.append(loss.item())
                val_d_pos.append(d_pos)
                val_d_neg.append(d_neg)
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(epoch_losses))
        print(f"== epoch {epoch} done: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_D_pos={np.mean(val_d_pos):.4f} val_D_neg={np.mean(val_d_neg):.4f} ==", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_D_pos": float(np.mean(val_d_pos)), "val_D_neg": float(np.mean(val_d_neg))})
        with open(out_dir / "training_curves.json", "w") as f:
            json.dump(history, f)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), out_dir / "outfit_transformer_siglip_best.pt")
            print(f"  -> new best val_loss {val_loss:.4f}, checkpoint saved", flush=True)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
                break

    return history

"""
Phase 13b: training loop for CSA-Net's subspace attention on a frozen SigLIP
backbone. Same sample construction and loss as phase 13's train_core.py, but
much simpler: no image loading, no gradient accumulation/micro-batching (no
CNN forward+backward means no OOM risk -- a full 96-outfit logical batch's
worth of SigLIP lookups is a trivial matrix operation, not ~1,500 images
through ResNet18), no backbone freeze/unfreeze schedule (there's no backbone
to freeze here -- see ../architecture_notes.md).

Reuses phase 13's training data and mined negative candidates directly
(backbone-independent -- see architecture_notes.md), and phase 9's
precomputed SigLIP embeddings (also backbone-independent, since this IS the
backbone here).
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CSANetSigLIP, pairwise_distance, outfit_ranking_loss, uniformity_loss, NUM_CATEGORIES

NUM_NEGATIVES = 10  # matches phase 13's choice, for direct comparability
UNIFORMITY_WEIGHT = 1.0  # matches phase 13's choice


class TrainState:
    def __init__(self, embeddings_npz, training_data_path, negative_candidates_path, seed=0):
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
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.train_outfits = td["train_outfits"]
        self.val_outfits = td["val_outfits"]
        self.train_items_by_cat = td["train_items_by_category"]
        self.val_items_by_cat = td["val_items_by_category"]

        self.item_cat = {}
        for cat, items in self.train_items_by_cat.items():
            for i in items:
                self.item_cat[i] = cat
        for cat, items in self.val_items_by_cat.items():
            for i in items:
                self.item_cat[i] = cat

        with open(negative_candidates_path) as f:
            neg_cands = json.load(f)
        self.neg_candidates = {"train": neg_cands.get("train", {}), "val": neg_cands.get("val", {})}

    def _sample_negatives(self, positive_id, category, split, exclude, num_negatives):
        cands = self.neg_candidates[split].get(positive_id, [])
        cands = [c for c in cands if c not in exclude]
        if len(cands) >= num_negatives:
            return self.rng.sample(cands, num_negatives)
        pool = self.train_items_by_cat if split == "train" else self.val_items_by_cat
        extra_pool = [i for i in pool[category] if i not in exclude and i not in cands]
        n_extra = num_negatives - len(cands)
        extra = self.rng.sample(extra_pool, min(n_extra, len(extra_pool))) if extra_pool else []
        return cands + extra

    def make_sample(self, outfit_record, split, num_negatives=NUM_NEGATIVES):
        items = outfit_record["items"]
        positive = self.rng.choice(items)
        context = [i for i in items if i != positive]
        pos_cat = self.item_cat[positive]
        exclude = set(items)
        negatives = self._sample_negatives(positive, pos_cat, split, exclude, num_negatives)
        return {
            "context_items": context,
            "context_cats": [self.item_cat[i] for i in context],
            "positive": positive,
            "positive_cat": pos_cat,
            "negatives": negatives,
        }

    def onehot(self, cat_name, batch=1):
        v = torch.zeros(batch, len(self.categories))
        v[:, self.cat_to_idx[cat_name]] = 1.0
        return v


def build_batch_vectors(state, samples, device):
    unique_ids = []
    seen = set()
    for s in samples:
        for i in s["context_items"] + [s["positive"]] + s["negatives"]:
            if i not in seen:
                seen.add(i)
                unique_ids.append(i)
    gidx = [state.idx[i] for i in unique_ids]
    vecs = torch.tensor(state.embeddings[gidx], device=device)  # (N_unique, 768)
    id_to_pos = {i: p for p, i in enumerate(unique_ids)}
    return vecs, id_to_pos


def compute_batch_loss(model, state, samples, x_all, id_to_pos, device, aggregation="min"):
    """Identical logic to phase 13's compute_batch_loss (see that file for
    the full explanation of the per-context-item scoring and the uniformity
    term); only the variable name `x_all` now holds SigLIP-derived features
    instead of ResNet18-derived ones."""
    losses = []
    d_pos_list, d_neg_list = [], []
    representative_embeddings = []
    for s in samples:
        n_ctx = len(s["context_items"])
        if n_ctx == 0:
            continue
        ctx_pos = [id_to_pos[i] for i in s["context_items"]]
        x_ctx = x_all[ctx_pos]
        cat_s_ctx = torch.cat([state.onehot(c) for c in s["context_cats"]], dim=0).to(device)
        cat_t = state.onehot(s["positive_cat"], batch=n_ctx).to(device)
        f_ctx = model.embed_from_feature(x_ctx, cat_s_ctx, cat_t)

        x_pos = x_all[id_to_pos[s["positive"]]].unsqueeze(0).expand(n_ctx, -1)
        f_pos = model.embed_from_feature(x_pos, cat_s_ctx, cat_t)
        d_pos_i = pairwise_distance(f_ctx, f_pos)
        D_pos = d_pos_i.mean()
        representative_embeddings.append(F.normalize(f_pos.mean(dim=0), p=2, dim=-1))

        d_negs = []
        for neg_id in s["negatives"]:
            x_neg = x_all[id_to_pos[neg_id]].unsqueeze(0).expand(n_ctx, -1)
            f_neg = model.embed_from_feature(x_neg, cat_s_ctx, cat_t)
            d_neg_i = pairwise_distance(f_ctx, f_neg)
            d_negs.append(d_neg_i.mean())
        D_negs = torch.stack(d_negs)

        loss = outfit_ranking_loss(D_pos.unsqueeze(0), D_negs.unsqueeze(0), aggregation=aggregation)
        losses.append(loss)
        d_pos_list.append(D_pos.item())
        d_neg_list.append(D_negs.min().item() if aggregation == "min" else D_negs.mean().item())

    ranking_loss = torch.stack(losses).mean()
    uniformity = uniformity_loss(torch.stack(representative_embeddings))
    total_loss = ranking_loss + UNIFORMITY_WEIGHT * uniformity
    return total_loss, float(np.mean(d_pos_list)), float(np.mean(d_neg_list))


def run_training(embeddings_npz, training_data_path, negative_candidates_path,
                  out_dir, device, max_epochs, batch_size, lr, patience,
                  n_train_outfits=None, n_val_outfits=None, log_every=20,
                  num_negatives=NUM_NEGATIVES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState(embeddings_npz, training_data_path, negative_candidates_path, seed=0)
    train_outfits = state.train_outfits[:n_train_outfits] if n_train_outfits else state.train_outfits
    val_outfits = state.val_outfits[:n_val_outfits] if n_val_outfits else state.val_outfits

    model = CSANetSigLIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = max_epochs * max(1, len(train_outfits) // batch_size)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    val_state = TrainState(embeddings_npz, training_data_path, negative_candidates_path, seed=42)
    val_samples_fixed = [val_state.make_sample(o, "val", num_negatives) for o in val_outfits]

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
            samples = [state.make_sample(train_outfits[i], "train", num_negatives) for i in b_idx]
            vecs, id_to_pos = build_batch_vectors(state, samples, device)

            optimizer.zero_grad()
            x_all = model.encode_feature(vecs)
            loss, d_pos, d_neg = compute_batch_loss(model, state, samples, x_all, id_to_pos, device)
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
                vecs, id_to_pos = build_batch_vectors(val_state, batch, device)
                x_all = model.encode_feature(vecs)
                loss, d_pos, d_neg = compute_batch_loss(model, val_state, batch, x_all, id_to_pos, device)
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
            torch.save(model.state_dict(), out_dir / "csa_net_siglip_best.pt")
            print(f"  -> new best val_loss {val_loss:.4f}, checkpoint saved", flush=True)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
                break

    return history

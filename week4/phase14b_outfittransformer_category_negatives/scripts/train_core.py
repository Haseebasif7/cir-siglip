"""
Phase 14b: training loop for OutfitTransformer's set-encoder mechanism,
identical to phase 14 except for WHERE the triplet loss's negative comes
from, per the diagnosed problem in
../../phase14b_outfittransformer_category_negatives.md.

Phase 14 used `InBatchTripletMarginLoss` exactly as the reference repo does:
the hardest negative for anchor i is the closest OTHER outfit's target
already present in the same training batch, with no category restriction.
Directly measured in phase 14 (`phase14_notes.md`): only 13.1% of possible
in-batch negative pairs share a category, while the real CIR evaluation
task restricts every candidate pool to the target's own category. That
mismatch was phase 14's diagnosed cause for scoring below raw untrained
SigLIP despite a well-converged training loss.

Fix here: for each sample, draw NUM_NEGATIVES negatives from a same-category
candidate pool instead of using other outfits' in-batch targets. This
mirrors CSA-Net's own successful negative-sampling machinery
(week4/phase13_csa_net_baseline/scripts/train_core.py, `_sample_negatives`)
almost exactly -- same source data even: phase 13's own precomputed
same-category, SigLIP-similarity-mined candidate list
(`week4/phase13_csa_net_baseline/data/negative_candidates.json`), built from
the identical item universe (phase 9's siglip_base.npz) and the identical
train/val split (phase 13's training_data.json, already reused directly by
phase 14 for its own outfit lists) -- reused here rather than re-mined,
since re-mining would just reproduce the same category-restricted neighbor
lists phase 13 already computed and validated. Falls back to a random
same-category item (also exactly CSA-Net's own fallback rule) when a target
has fewer than NUM_NEGATIVES mined candidates.

The architecture, the transformer forward passes (embed_query /
embed_item_alone), and the triplet-margin comparison itself (positive
distance vs. hardest-negative distance, relu(pos - hardest_neg + margin))
are otherwise unchanged from phase 14 -- only the population the hardest
negative is chosen FROM has changed.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import OutfitTransformerSigLIP, uniformity_loss

UNIFORMITY_WEIGHT = 1.0
NUM_NEGATIVES = 10  # matches CSA-Net's own NUM_NEGATIVES (phase13/scripts/train_core.py) exactly, per the brief's instruction to mirror that already-successful practice


class TrainState:
    def __init__(self, embeddings_npz, training_data_path, negative_candidates_path, seed=0, negative_mode="mined"):
        self.rng = random.Random(seed)
        self.negative_mode = negative_mode  # "mined" (SigLIP-nearest same-category, see below) or "random" (uniform same-category, no mining) -- see run 2's ablation in ../training_log.md for why this was added

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
        self.train_items_by_cat = td["train_items_by_category"]
        self.val_items_by_cat = td["val_items_by_category"]

        # item_id -> category, needed to look up each sample's target category
        # -- exactly phase 13's own item_cat construction (train_core.py)
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

    def _sample_negatives(self, target_id, category, split, exclude, num_negatives):
        """negative_mode="mined": exactly CSA-Net's own _sample_negatives
        (phase13/scripts/train_core.py) -- mined same-category candidates
        (SigLIP-nearest neighbors, capped at 0.97 similarity so not
        near-duplicates, but still each anchor's closest same-category
        items) first, topped up with random same-category items if the
        mined list is short.

        negative_mode="random": skips the mined candidate list entirely,
        draws all num_negatives uniformly at random from the target's own
        category pool. Added after run 1 (mined) converged with the
        triplet-margin term stuck unresolved the whole run (D_pos > D_neg
        throughout, see ../training_log.md) and scored far below even
        phase 14's broken in-batch baseline -- this ablation tests whether
        that failure is caused by negative DIFFICULTY (mined near-neighbors
        being too hard for this small transformer's triplet loss to
        separate) rather than by the category restriction itself, following
        this project's own established phase 7-9 finding that hard-negative
        mining underperforms for its embedding losses.

        Both modes return negatives restricted to `category` by construction
        -- the category-match-rate fix (verify_negative_sampling.py) applies
        identically to either."""
        if self.negative_mode == "mined":
            cands = self.neg_candidates[split].get(target_id, [])
            cands = [c for c in cands if c not in exclude]
        else:
            cands = []
        if len(cands) >= num_negatives:
            return self.rng.sample(cands, num_negatives)
        pool = self.train_items_by_cat if split == "train" else self.val_items_by_cat
        extra_pool = [i for i in pool[category] if i not in exclude and i not in cands]
        n_extra = num_negatives - len(cands)
        extra = self.rng.sample(extra_pool, min(n_extra, len(extra_pool))) if extra_pool else []
        return cands + extra

    def make_sample(self, outfit_record, split, num_negatives=NUM_NEGATIVES):
        """Random target item + rest-of-outfit context (same as phase 14's
        make_sample), plus this phase's new same-category negative set."""
        items = outfit_record["items"]
        target = self.rng.choice(items)
        context = [i for i in items if i != target]
        target_cat = self.item_cat[target]
        exclude = set(items)
        negatives = self._sample_negatives(target, target_cat, split, exclude, num_negatives)
        return {"context_items": context, "target": target, "target_cat": target_cat, "negatives": negatives}


def build_batch(state, samples, device):
    """Builds padded context-token tensors, target-item tensors, and
    negative-item tensors for a batch of samples. Negatives are padded to a
    uniform count per batch (usually exactly NUM_NEGATIVES, shorter only if
    a target's category pool itself has too few items to fill it -- tracked
    via neg_mask so short rows don't contribute phantom zero-vector
    negatives to the loss)."""
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

    Mmax = max(len(s["negatives"]) for s in samples)
    Mmax = max(Mmax, 1)
    neg_vecs = np.zeros((B, Mmax, state.embeddings.shape[1]), dtype=np.float32)
    neg_mask = np.zeros((B, Mmax), dtype=bool)  # True = valid negative
    for i, s in enumerate(samples):
        for j, item in enumerate(s["negatives"]):
            neg_vecs[i, j] = state.embeddings[state.idx[item]]
            neg_mask[i, j] = True

    return (
        torch.tensor(ctx_vecs, device=device),
        torch.tensor(ctx_mask, device=device),
        torch.tensor(target_vecs, device=device),
        torch.tensor(neg_vecs, device=device),
        torch.tensor(neg_mask, device=device),
    )


def category_negative_triplet_loss(query_emb, answer_emb, neg_emb, neg_mask, margin):
    """Same comparison as phase 14's in_batch_triplet_loss (positive
    distance vs. hardest-negative distance, relu(pos - hardest_neg +
    margin)), but the hardest negative is now chosen from this sample's OWN
    same-category candidate set (neg_emb, neg_mask), not from other outfits'
    targets elsewhere in the batch. query_emb/answer_emb: (B, D). neg_emb:
    (B, M, D). neg_mask: (B, M) bool, True=valid. Returns (loss, mean
    positive distance, mean hardest-negative distance)."""
    pos = torch.norm(query_emb - answer_emb, p=2, dim=-1)  # (B,)
    d_negs = torch.norm(query_emb.unsqueeze(1) - neg_emb, p=2, dim=-1)  # (B, M)
    d_negs = d_negs.masked_fill(~neg_mask, float("inf"))
    hardest_neg, _ = d_negs.min(dim=1)  # (B,)
    loss = F.relu(pos - hardest_neg + margin)
    return loss.mean(), pos.mean().item(), hardest_neg.mean().item()


def compute_batch_loss(model, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, margin, use_uniformity):
    B, Lmax, D = ctx_vecs.shape
    ctx_tokens = model.encode_item_tokens(ctx_vecs.view(B * Lmax, D)).view(B, Lmax, -1)
    query_emb = model.embed_query(ctx_tokens, ctx_mask)

    target_tokens = model.encode_item_tokens(target_vecs)
    answer_emb = model.embed_item_alone(target_tokens)

    Bn, M, Dn = neg_vecs.shape
    neg_tokens = model.encode_item_tokens(neg_vecs.view(Bn * M, Dn)).view(Bn, M, -1)
    neg_emb = model.embed_item_alone(neg_tokens.view(Bn * M, -1)).view(Bn, M, -1)

    triplet_loss, d_pos, d_neg = category_negative_triplet_loss(query_emb, answer_emb, neg_emb, neg_mask, margin)
    total_loss = triplet_loss
    if use_uniformity:
        total_loss = total_loss + UNIFORMITY_WEIGHT * uniformity_loss(answer_emb)
    return total_loss, d_pos, d_neg


def run_training(embeddings_npz, training_data_path, negative_candidates_path, out_dir, device,
                  max_epochs, batch_size, lr, patience, margin,
                  n_train_outfits=None, n_val_outfits=None, log_every=50,
                  use_uniformity=True, num_negatives=NUM_NEGATIVES, negative_mode="mined",
                  checkpoint_name="outfit_transformer_siglip_category_neg_best.pt"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState(embeddings_npz, training_data_path, negative_candidates_path, seed=0, negative_mode=negative_mode)
    train_outfits = [o for o in state.train_outfits if len(o["items"]) >= 2]
    val_outfits = [o for o in state.val_outfits if len(o["items"]) >= 2]
    if n_train_outfits:
        train_outfits = train_outfits[:n_train_outfits]
    if n_val_outfits:
        val_outfits = val_outfits[:n_val_outfits]

    model = OutfitTransformerSigLIP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = max(1, math.ceil(len(train_outfits) / batch_size))
    total_steps = max_epochs * steps_per_epoch + 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )

    val_state = TrainState(embeddings_npz, training_data_path, negative_candidates_path, seed=42, negative_mode=negative_mode)
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
            if len(b_idx) < 2:
                continue
            samples = [state.make_sample(train_outfits[i], "train", num_negatives) for i in b_idx]
            ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask = build_batch(state, samples, device)

            optimizer.zero_grad()
            loss, d_pos, d_neg = compute_batch_loss(
                model, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, margin, use_uniformity
            )
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
                ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask = build_batch(val_state, batch, device)
                loss, d_pos, d_neg = compute_batch_loss(
                    model, ctx_vecs, ctx_mask, target_vecs, neg_vecs, neg_mask, margin, use_uniformity
                )
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
            torch.save(model.state_dict(), out_dir / checkpoint_name)
            print(f"  -> new best val_loss {val_loss:.4f}, checkpoint saved", flush=True)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
                break

    return history

"""
Phase 13, step 3: CSA-Net training loop core logic, shared by the local
smoke test (train_smoke_test.py) and the Modal full-scale run
(modal_train_csa_net.py) -- same code path in both, only the image
directory / dataset size / epoch count differ, so a passing smoke test is a
real guarantee about the Modal run's correctness, not a separate reimplementation.

Design (see model.py docstring + ../implementation_notes.md for the full
paper trace):
  - one "training sample" = one outfit, one held-out positive item (context
    = the rest), M same-category negatives drawn from the precomputed
    mined-candidate list (data/negative_candidates.json)
  - a batch = B such samples. All UNIQUE images needed across the batch
    (context + positive + negatives, deduplicated) are loaded once and run
    through the CNN backbone in a SINGLE forward pass -- the expensive part
    happens once per unique image, not once per (image, context-item) pair
  - per-outfit distances (eq. 5 of the paper) are then computed from that
    shared feature tensor via cheap tensor ops, looped in Python once per
    sample in the batch (no CNN work in this loop)
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMAGE_LOAD_WORKERS = 16  # PIL decode/resize releases the GIL for most of its
                          # work, so threads give a real speedup here; image
                          # loading (not the CNN) was the actual bottleneck
                          # observed in early testing (~1,500 images/step).

from model import CSANet, pairwise_distance, outfit_ranking_loss, uniformity_loss, NUM_CATEGORIES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

NUM_NEGATIVES = 10  # paper doesn't state the exact negative count per sample;
                     # 10 chosen as a reasonable, documented default (see
                     # implementation_notes.md) balancing fidelity to
                     # "semi-hard mining over a candidate set" against compute
                     # budget -- large enough for a real min-aggregated hinge
                     # signal, small enough to keep per-step image count
                     # tractable (~B * (avg context + 1 + 10) images/step).


class TrainState:
    """Holds everything needed to build training batches: outfit records,
    category vocab, image loader, mined negative candidates."""

    def __init__(self, images_dir, training_data_path, negative_candidates_path,
                 device, seed=0):
        self.images_dir = Path(images_dir)
        self.device = device
        self.rng = random.Random(seed)

        with open(training_data_path) as f:
            td = json.load(f)
        self.categories = td["categories"]
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.train_outfits = td["train_outfits"]
        self.val_outfits = td["val_outfits"]
        self.train_items_by_cat = td["train_items_by_category"]
        self.val_items_by_cat = td["val_items_by_category"]

        # item_id -> category, needed to look up context/negative categories
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
        # Fallback: not enough mined candidates -- top up with random
        # same-category items (documented in implementation_notes.md).
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

    def load_image(self, item_id, train):
        path = self.images_dir / f"{item_id}.jpg"
        img = Image.open(path).convert("RGB")
        t = TRAIN_TRANSFORM if train else EVAL_TRANSFORM
        return t(img)


def build_batch_images(state, samples, train):
    """Dedup + load all images needed across a batch of samples. Returns
    (image_tensor (N,3,224,224), id_to_pos dict)."""
    unique_ids = []
    seen = set()
    for s in samples:
        for i in s["context_items"] + [s["positive"]] + s["negatives"]:
            if i not in seen:
                seen.add(i)
                unique_ids.append(i)
    with ThreadPoolExecutor(max_workers=IMAGE_LOAD_WORKERS) as pool:
        loaded = list(pool.map(lambda i: state.load_image(i, train), unique_ids))
    imgs = torch.stack(loaded)
    id_to_pos = {i: p for p, i in enumerate(unique_ids)}
    return imgs, id_to_pos


UNIFORMITY_WEIGHT = 1.0  # see model.uniformity_loss's docstring -- added after
                          # four independent real-data runs all showed the
                          # same representation-collapse failure mode


def compute_batch_loss(model, state, samples, x_all, id_to_pos, device, aggregation="min"):
    """x_all: (N_unique, D) base CNN features for the whole batch (already on
    device). Returns scalar mean loss over the batch's samples, plus
    diagnostics (mean D_pos, mean D_neg_agg) for logging."""
    losses = []
    d_pos_list, d_neg_list = [], []
    representative_embeddings = []  # one per sample, for the uniformity term
    for s in samples:
        n_ctx = len(s["context_items"])
        if n_ctx == 0:
            continue
        ctx_pos = [id_to_pos[i] for i in s["context_items"]]
        x_ctx = x_all[ctx_pos]  # (n_ctx, D)
        cat_s_ctx = torch.cat([state.onehot(c) for c in s["context_cats"]], dim=0).to(device)
        cat_t = state.onehot(s["positive_cat"], batch=n_ctx).to(device)
        f_ctx = model.embed_from_feature(x_ctx, cat_s_ctx, cat_t)  # f_i^o, (n_ctx, D)

        x_pos = x_all[id_to_pos[s["positive"]]].unsqueeze(0).expand(n_ctx, -1)
        f_pos = model.embed_from_feature(x_pos, cat_s_ctx, cat_t)  # f_i^P
        d_pos_i = pairwise_distance(f_ctx, f_pos)  # (n_ctx,)
        D_pos = d_pos_i.mean()
        representative_embeddings.append(F.normalize(f_pos.mean(dim=0), p=2, dim=-1))

        d_negs = []
        for neg_id in s["negatives"]:
            x_neg = x_all[id_to_pos[neg_id]].unsqueeze(0).expand(n_ctx, -1)
            f_neg = model.embed_from_feature(x_neg, cat_s_ctx, cat_t)
            d_neg_i = pairwise_distance(f_ctx, f_neg)
            d_negs.append(d_neg_i.mean())
        D_negs = torch.stack(d_negs)  # (M,)

        loss = outfit_ranking_loss(D_pos.unsqueeze(0), D_negs.unsqueeze(0), aggregation=aggregation)
        losses.append(loss)
        d_pos_list.append(D_pos.item())
        d_neg_list.append(D_negs.min().item() if aggregation == "min" else D_negs.mean().item())

    ranking_loss = torch.stack(losses).mean()
    uniformity = uniformity_loss(torch.stack(representative_embeddings))
    total_loss = ranking_loss + UNIFORMITY_WEIGHT * uniformity
    return total_loss, float(np.mean(d_pos_list)), float(np.mean(d_neg_list))


def run_training(images_dir, training_data_path, negative_candidates_path,
                  out_dir, device, max_epochs, batch_size, lr, patience,
                  n_train_outfits=None, n_val_outfits=None, log_every=20,
                  num_negatives=NUM_NEGATIVES, num_workers_note="",
                  micro_batch_size=12, freeze_backbone_epochs=3):
    """Full training loop. n_train_outfits/n_val_outfits cap the dataset size
    (used by the local smoke test; None = use everything, the real run).

    micro_batch_size: a `batch_size`-outfit step (96, the paper's own value)
    needs ~1,500 unique images through ResNet18 WITH gradients at once --
    this OOM'd a 24GB A10G in initial testing (tried to allocate >20GB for a
    single forward+backward). Fixed via gradient accumulation: each logical
    batch of `batch_size` outfits is split into `micro_batch_size`-outfit
    chunks, each chunk's loss is scaled by its share of the batch and
    backward()-accumulated, with a single optimizer.step() after the whole
    batch -- mathematically the same averaged gradient the paper's batch=96
    would produce, just computed with bounded peak memory (~12 outfits'
    worth of unique images resident at a time, not 96's).

    freeze_backbone_epochs: a SECOND, more severe collapse mode was found
    even after adding L2-normalization (see model.py) -- on the full
    251k-item dataset (not just the small local smoke test), D_pos and
    D_neg still shrank toward 0 together across a full epoch on real Modal
    GPU runs, converging the loss to exactly the margin value. Since
    normalized embeddings can't collapse in MAGNITUDE anymore, this has to
    be DIRECTION collapse: the still-randomly-initialized proj/attention/
    mask layers give a strong, consistent early gradient that's easiest to
    satisfy by driving the whole (still fine-tuning) ResNet18 backbone
    toward outputting near-constant features regardless of input, rather
    than learning real per-item structure. Standard, well-established fix
    (not paper-specified, but common transfer-learning practice, not a
    change to the loss/architecture itself): freeze the pretrained backbone
    entirely for the first few epochs, letting only the new proj/attention/
    mask layers train against STABLE, already-somewhat-discriminative
    ImageNet features, before unfreezing the backbone for full end-to-end
    fine-tuning. Backbone BatchNorm layers are also held in eval() mode
    while frozen so their running statistics don't drift either."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState(images_dir, training_data_path, negative_candidates_path,
                        device, seed=0)
    train_outfits = state.train_outfits[:n_train_outfits] if n_train_outfits else state.train_outfits
    val_outfits = state.val_outfits[:n_val_outfits] if n_val_outfits else state.val_outfits

    model = CSANet(pretrained=True).to(device)
    if freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"Backbone frozen for the first {freeze_backbone_epochs} epoch(s).", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # paper: ADAM, initial lr 5e-5
    total_steps = max_epochs * max(1, len(train_outfits) // batch_size)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )  # paper: "linearly decreases the lr to zero", no warmup

    # Fixed validation sample set, sampled ONCE (not re-randomized per epoch)
    # so the validation curve reflects real training progress, not
    # per-epoch negative-sampling noise.
    val_rng = random.Random(42)
    val_state = TrainState(images_dir, training_data_path, negative_candidates_path, device, seed=42)
    val_samples_fixed = [val_state.make_sample(o, "val", num_negatives) for o in val_outfits]

    history = []
    best_val_loss = float("inf")
    epochs_since_improve = 0
    step = 0

    for epoch in range(max_epochs):
        model.train()
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print(f"Unfroze backbone at epoch {epoch}.", flush=True)
        if epoch < freeze_backbone_epochs:
            model.backbone.eval()  # keep BN running stats fixed too while frozen
        order = list(range(len(train_outfits)))
        state.rng.shuffle(order)
        epoch_losses = []
        for b_start in range(0, len(order), batch_size):
            b_idx = order[b_start:b_start + batch_size]
            optimizer.zero_grad()
            loss_sum, d_pos_sum, d_neg_sum, n_seen = 0.0, 0.0, 0.0, 0
            for m_start in range(0, len(b_idx), micro_batch_size):
                m_idx = b_idx[m_start:m_start + micro_batch_size]
                samples = [state.make_sample(train_outfits[i], "train", num_negatives) for i in m_idx]
                imgs, id_to_pos = build_batch_images(state, samples, train=True)
                imgs = imgs.to(device)

                x_all = model.encode_image(imgs)
                loss, d_pos, d_neg = compute_batch_loss(model, state, samples, x_all, id_to_pos, device)
                (loss * (len(m_idx) / len(b_idx))).backward()  # accumulate, scaled to average over full batch

                loss_sum += loss.item() * len(m_idx)
                d_pos_sum += d_pos * len(m_idx)
                d_neg_sum += d_neg * len(m_idx)
                n_seen += len(m_idx)
            optimizer.step()
            scheduler.step()

            batch_loss = loss_sum / n_seen
            epoch_losses.append(batch_loss)
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}: loss={batch_loss:.4f} "
                      f"D_pos={d_pos_sum/n_seen:.4f} D_neg={d_neg_sum/n_seen:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        # Validation pass, fixed sample set
        model.eval()
        val_losses, val_d_pos, val_d_neg = [], [], []
        with torch.no_grad():
            for b_start in range(0, len(val_samples_fixed), micro_batch_size):
                batch = val_samples_fixed[b_start:b_start + micro_batch_size]
                imgs, id_to_pos = build_batch_images(val_state, batch, train=False)
                imgs = imgs.to(device)
                x_all = model.encode_image(imgs)
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
            torch.save(model.state_dict(), out_dir / "csa_net_best.pt")
            print(f"  -> new best val_loss {val_loss:.4f}, checkpoint saved", flush=True)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
                break

    return history

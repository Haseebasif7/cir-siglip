"""
Phase 30: Modal GPU training app for the multi-aspect item representation
(K=4 independently-normalized 32-d aspect vectors per item, plus a learned
query-side softmax aggregation layer, replacing phase 27/28's single 128-d
projection). Reuses phase 27's already-deployed data volume
("phase27-text-category-data" -- siglip_base.npz, text_embeddings.npz,
positive_edges.json, cir_val_benchmark.json, cir_test_benchmark.json) as-is,
nothing new to upload: this phase changes only the projection head's
architecture, not any input data. Own app ("phase30-multi-aspect"), own
function, so runs stay cleanly separated from phase 27/28/29's.

Per the brief: base_repr = normalize(concat(image, text)), 1536-d, IDENTICAL
to phase 27/28's text_only (use_category="none") input construction -- this
phase's only variable is what happens to that 1536-d vector after the shared
trunk (multi-aspect heads + query-side aggregation instead of a single
1024->128 head). See model.py for the architecture itself.

During TRAINING, the anchor plays the query role (context size exactly 1),
mirroring phase 27/28/29's own precedent: aggregation weights are computed
from the anchor's own aspect vectors, and the positive / extra negatives are
scored against it via the weighted per-aspect cosine sum. At EVALUATION time,
the query is a genuine mean-pooled multi-item context (see
evaluate_recall_gpu): each aspect is mean-pooled independently across context
items, then renormalized, before the aggregation layer and per-aspect scoring
run -- this generalizes the training procedure exactly the way phase 27/28's
mean-pooled query at eval time generalizes their single-item training anchor,
so there is no train/eval mismatch of the kind diagnosed in phase 29 (this
phase's aggregation weights are query-only, never candidate-conditioned, so
nothing here depends on which candidate is being scored).
"""
import time

import modal

app = modal.App("phase30-multi-aspect")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
volume = modal.Volume.from_name("phase27-text-category-data", create_if_missing=False)

DATA_DIR = "/data"
CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]

# Same tuned hyperparameters as phase 27/28's text_only, held fixed per the
# brief -- only the projection architecture varies in this phase's runs.
DEFAULT_CONFIG = {
    "name": "default",
    "n_aspects": 4,
    "aspect_dim": 32,
    "hidden_dim": 1024,
    "lr": 0.001,
    "batch_size": 256,
    "weight_decay": 0.0,
    "tau": 0.15,
    "r_neg": 8,
    "seed": 42,
    "max_epochs": 12,
    "patience": 4,
    "min_delta": 0.0005,
    "eval_every": 1,
    "save_checkpoint": False,
}


def _build_model_and_loss():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiAspectProjectionHead(nn.Module):
        def __init__(self, in_dim=1536, hidden_dim=1024, n_aspects=4, aspect_dim=32, dropout=0.1):
            super().__init__()
            self.n_aspects = n_aspects
            self.aspect_dim = aspect_dim
            self.trunk = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.aspect_heads = nn.ModuleList([nn.Linear(hidden_dim, aspect_dim) for _ in range(n_aspects)])
            self.agg_layer = nn.Linear(n_aspects * aspect_dim, n_aspects)

        def forward(self, x):
            h = self.trunk(x)
            aspects = [F.normalize(head(h), p=2, dim=-1) for head in self.aspect_heads]
            return torch.stack(aspects, dim=-2)

        def agg_weights(self, aspect_vecs):
            flat = aspect_vecs.flatten(start_dim=-2)
            return F.softmax(self.agg_layer(flat), dim=-1)

    def inbatch_scores(query_aspects, query_weights, pool_aspects):
        sim_k = torch.einsum("bkd,pkd->bpk", query_aspects, pool_aspects)
        return torch.einsum("bpk,bk->bp", sim_k, query_weights)

    def aspect_score_extra(query_aspects, query_weights, extra_aspects):
        sim_k = torch.einsum("bkd,brkd->brk", query_aspects, extra_aspects)
        return torch.einsum("brk,bk->br", sim_k, query_weights)

    def mnrl_loss_multiaspect(anchor_aspects, anchor_weights, positive_aspects, extra_aspects, mask, tau):
        inbatch = inbatch_scores(anchor_aspects, anchor_weights, positive_aspects) / tau
        inbatch = inbatch.masked_fill(mask, float("-inf"))
        extra = aspect_score_extra(anchor_aspects, anchor_weights, extra_aspects) / tau
        logits = torch.cat([inbatch, extra], dim=1)
        labels = torch.arange(anchor_aspects.shape[0], device=anchor_aspects.device)
        return F.cross_entropy(logits, labels)

    return MultiAspectProjectionHead, mnrl_loss_multiaspect, inbatch_scores


# max_containers=5, inherited from phase 28/29's own throttling fix after the
# m-haseebasif5 workspace's GPU concurrency quota was hit by an unthrottled
# .map() -- carried forward as standing practice for every phase since.
@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=7200, max_containers=5)
def train_one(config: dict) -> dict:
    import json
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    cfg = {**DEFAULT_CONFIG, **config}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MultiAspectProjectionHead, mnrl_loss_multiaspect, inbatch_scores = _build_model_and_loss()
    data_dir = Path(DATA_DIR)

    # --- Credit-safety: idempotent skip + warm-start resume, unchanged
    # pattern from phase 27/28/29. ---
    status_path = data_dir / f"status_{cfg['name']}.json"
    ckpt_path = data_dir / f"checkpoint_{cfg['name']}.pt"

    if status_path.exists():
        with open(status_path) as f:
            prior_status = json.load(f)
        if prior_status.get("status") == "done":
            print(f"{cfg['name']}: already done (val_recall10="
                  f"{prior_status.get('best_recall10')}), skipping -- credit-safety short-circuit.")
            return prior_status["result"]

    # --- Load base representation: image+text concat, 1536-d, identical to
    # phase 27/28's text_only construction. ---
    img_npz = np.load(data_dir / "siglip_base.npz", allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    n_items = len(item_ids)

    image_emb = img_npz["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_npz = np.load(data_dir / "text_embeddings.npz", allow_pickle=True)
    text_ids = [str(a) for a in text_npz["item_ids"]]
    assert text_ids == item_ids, "text_embeddings.npz must be positionally aligned to siglip_base.npz"
    text_emb = text_npz["embeddings"].astype(np.float32)

    torch.manual_seed(cfg["seed"])

    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)  # (n_items, 1536)

    # --- Training/validation edges, identical source/split to phase 9/23/25/26/27/28. ---
    with open(data_dir / "positive_edges.json") as f:
        edge_records = json.load(f)
    train_pairs = [(e["source"], e["target"]) for e in edge_records if e["split"] == "train"]
    val_pairs = [(e["source"], e["target"]) for e in edge_records if e["split"] == "val"]
    train_edges = np.array([(id_to_gidx[a], id_to_gidx[b]) for a, b in train_pairs], dtype=np.int64)
    val_edges = np.array([(id_to_gidx[a], id_to_gidx[b]) for a, b in val_pairs], dtype=np.int64)

    positive_sets = {}
    for a_idx, b_idx in np.concatenate([train_edges, val_edges], axis=0):
        positive_sets.setdefault(int(a_idx), set()).add(int(b_idx))

    with open(data_dir / "cir_val_benchmark.json") as f:
        bench = json.load(f)
    val_pools, val_queries = bench["pools"], bench["queries"]

    model = MultiAspectProjectionHead(
        in_dim=1536, hidden_dim=cfg["hidden_dim"],
        n_aspects=cfg["n_aspects"], aspect_dim=cfg["aspect_dim"],
    ).to(device)

    def evaluate_recall_gpu(ks=(10, 30, 50)):
        model.eval()
        entropies_raw, entropies_norm = [], []
        with torch.no_grad():
            proj_aspects = model(base_repr)  # (n_items, K, D)

            by_cat = defaultdict(list)
            for qi, q in enumerate(val_queries):
                by_cat[q["category"]].append(qi)

            hits = {k: 0 for k in ks}
            n_total, n_skipped = 0, 0
            for cat, qidxs in by_cat.items():
                if cat not in val_pools:
                    n_skipped += len(qidxs)
                    continue
                pool_ids = val_pools[cat]
                pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
                pool_idx = [id_to_gidx[i] for i in pool_ids]
                pool_aspects = proj_aspects[pool_idx]  # (P, K, D)

                kept_qidx, kept_ctx_idx, tpos = [], [], []
                for qi in qidxs:
                    q = val_queries[qi]
                    ctx_idx = [id_to_gidx[i] for i in q["query_items"] if i in id_to_gidx]
                    if not ctx_idx or q["target_item"] not in pool_pos:
                        n_skipped += 1
                        continue
                    kept_qidx.append(qi)
                    kept_ctx_idx.append(ctx_idx)
                    tpos.append(pool_pos[q["target_item"]])
                if not kept_qidx:
                    continue

                # Per-aspect mean-pool over context items, then renormalize
                # each aspect independently -- the direct multi-item
                # generalization of the single-item training anchor.
                q_aspects_list = []
                for ctx_idx in kept_ctx_idx:
                    v = proj_aspects[ctx_idx].mean(dim=0)  # (K, D)
                    v = F.normalize(v, p=2, dim=-1)
                    q_aspects_list.append(v)
                q_aspects = torch.stack(q_aspects_list)  # (nq, K, D)
                q_weights = model.agg_weights(q_aspects)  # (nq, K)
                entropies_raw.append((-(q_weights * (q_weights + 1e-9).log()).sum(dim=-1)))
                entropies_norm.append(entropies_raw[-1] / np.log(cfg["n_aspects"]))

                sims = inbatch_scores(q_aspects, q_weights, pool_aspects).cpu().numpy()  # (nq, P)
                tpos = np.array(tpos)
                tsims = sims[np.arange(len(kept_qidx)), tpos]
                ranks = (sims >= tsims[:, None]).sum(axis=1)
                n_total += len(kept_qidx)
                for k in ks:
                    hits[k] += int((ranks <= k).sum())

        recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
        mean_entropy_raw = torch.cat(entropies_raw).mean().item() if entropies_raw else 0.0
        mean_entropy_norm = torch.cat(entropies_norm).mean().item() if entropies_norm else 0.0
        return recall, n_total, n_skipped, mean_entropy_raw, mean_entropy_norm

    rng_np = np.random.default_rng(cfg["seed"])

    def sample_negatives(anchors_idx, R):
        B = len(anchors_idx)
        cand = rng_np.integers(0, n_items, size=(B, R))
        for i in range(B):
            a_i = int(anchors_idx[i])
            pos_i = positive_sets.get(a_i)
            for j in range(R):
                tries = 0
                while int(cand[i, j]) == a_i or (pos_i is not None and int(cand[i, j]) in pos_i):
                    cand[i, j] = rng_np.integers(0, n_items)
                    tries += 1
                    if tries > 20:
                        break
        return cand

    def build_inbatch_mask(anchors_idx, positives_idx):
        B = len(anchors_idx)
        mask = np.zeros((B, B), dtype=bool)
        for i in range(B):
            pos_i = positive_sets.get(int(anchors_idx[i]))
            if not pos_i:
                continue
            for j in range(B):
                if i != j and int(positives_idx[j]) in pos_i:
                    mask[i, j] = True
        return mask

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    B = cfg["batch_size"]
    R = cfg["r_neg"]
    tau = cfg["tau"]
    K = cfg["n_aspects"]

    best_recall10 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    if ckpt_path.exists():
        print(f"{cfg['name']}: found an existing checkpoint, warm-starting from it "
              f"(optimizer/epoch state is NOT restored, only weights).")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if status_path.exists():
            with open(status_path) as f:
                prior_status = json.load(f)
            best_recall10 = prior_status.get("best_recall10", -1.0) or -1.0
            print(f"  warm-started at best_recall10={best_recall10}")

    n_train = len(train_edges)
    order = np.arange(n_train)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        rng_np.shuffle(order)
        total_loss, n_batches = 0.0, 0
        train_entropies = []
        for start in range(0, n_train, B):
            batch_idx = order[start:start + B]
            if len(batch_idx) < 2:
                continue
            batch_edges = train_edges[batch_idx]
            anchors_idx = batch_edges[:, 0]
            positives_idx = batch_edges[:, 1]
            neg_idx = sample_negatives(anchors_idx, R)
            mask = torch.tensor(build_inbatch_mask(anchors_idx, positives_idx), device=device)

            anchors_t = torch.tensor(anchors_idx, device=device)
            positives_t = torch.tensor(positives_idx, device=device)
            neg_t = torch.tensor(neg_idx, device=device)

            a_repr = base_repr[anchors_t]
            p_repr = base_repr[positives_t]
            extra_repr = base_repr[neg_t]  # (B, R, 1536)

            anchor_aspects = model(a_repr)                  # (B, K, D)
            anchor_weights = model.agg_weights(anchor_aspects)  # (B, K)
            positive_aspects = model(p_repr)                # (B, K, D)
            Bb, Rr, D = extra_repr.shape
            extra_aspects = model(extra_repr.reshape(Bb * Rr, D)).reshape(Bb, Rr, K, cfg["aspect_dim"])

            loss = mnrl_loss_multiaspect(anchor_aspects, anchor_weights, positive_aspects, extra_aspects, mask, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                w = anchor_weights.detach()
                ent = -(w * (w + 1e-9).log()).sum(dim=-1)
                train_entropies.append(ent.mean().item())
        train_loss = total_loss / max(n_batches, 1)
        mean_train_entropy = float(np.mean(train_entropies)) if train_entropies else 0.0
        mean_train_entropy_norm = mean_train_entropy / np.log(K)

        do_eval = (epoch % cfg["eval_every"] == 0) or (epoch == cfg["max_epochs"] - 1)
        if do_eval:
            recall, n_total, n_skipped, val_ent, val_ent_norm = evaluate_recall_gpu()
            r10 = recall[10]
            curve.append({
                "epoch": epoch, "train_loss": train_loss, "recall10": recall[10],
                "recall30": recall[30], "recall50": recall[50],
                "train_agg_entropy": mean_train_entropy, "train_agg_entropy_norm": mean_train_entropy_norm,
                "val_agg_entropy": val_ent, "val_agg_entropy_norm": val_ent_norm,
            })
            improved = r10 > best_recall10 + cfg["min_delta"]
            if improved:
                best_recall10 = r10
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0

                if cfg.get("save_checkpoint"):
                    torch.save(best_state, ckpt_path)
                    with open(status_path, "w") as f:
                        json.dump({"status": "in_progress", "best_epoch": best_epoch,
                                    "best_recall10": best_recall10, "epoch_reached": epoch}, f)
                    volume.commit()
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break
        else:
            curve.append({"epoch": epoch, "train_loss": train_loss,
                           "train_agg_entropy": mean_train_entropy, "train_agg_entropy_norm": mean_train_entropy_norm})

    wall_time = time.time() - t_start

    result = {
        "config": cfg,
        "curve": curve,
        "best_epoch": best_epoch,
        "best_recall10": best_recall10,
        "n_epochs_run": len(curve),
        "wall_time_sec": wall_time,
        "device": device,
    }

    if cfg.get("save_checkpoint") and best_state is not None:
        torch.save(best_state, ckpt_path)
        result["checkpoint_path"] = str(ckpt_path)
        with open(status_path, "w") as f:
            json.dump({"status": "done", "best_epoch": best_epoch,
                        "best_recall10": best_recall10, "result": result}, f)
        volume.commit()

    return result


@app.local_entrypoint()
def smoke_test():
    """`modal run scripts/modal_app.py::smoke_test` -- 2-epoch sanity check
    before committing to the full single-seed gate run."""
    import json
    cfg = {"name": "smoke_multiaspect", "max_epochs": 2, "patience": 5, "eval_every": 1}
    result = train_one.remote(cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    print("curve:")
    for row in result["curve"]:
        print(" ", row)

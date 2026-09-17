"""
Phase 23: Modal GPU training app for the hyperparameter grid/sweeps this
phase requires. Local M4 Air was benchmarked at phase 9's original scale
(~6-7 min/epoch, single run) -- this phase needs dozens of full-dataset runs
across steps 3/4/5/6, which is squarely the "needs more compute than the
local M4 Air can reasonably provide" case the project's standing Modal
permission exists for. Using Modal's `.map()` also lets an entire grid run
as parallel containers instead of one sequential queue, which is the real
lever here (wall-clock, not just per-batch speed).

Trains the identical mechanism phase 9 used (ProjectionHead, MNRL loss,
in-batch false-negative masking, R random negatives per anchor, H=0 hard
negatives -- matching phase 9's own winning "model_a" configuration, the one
cited everywhere in this project) -- NOT touching architecture, scale, or
the frozen SigLIP backbone, per this phase's own "do not do yet" instruction.

Key methodological difference from phase 9's own 05_train_projection.py,
required by this phase's brief: model selection and early stopping are
keyed to validation-benchmark Recall@10 (checked every epoch), NOT
validation loss -- this project has repeatedly found val loss fails to
predict real Recall@K (phase 14, phase 14b, phase 16d), so this phase does
not repeat that mistake.

Usage:
  One-time data upload (run from local shell):
    modal volume create phase23-hp-tuning-data
    modal volume put phase23-hp-tuning-data \\
      ../../../week3/phase9_polyvore_compatibility/embeddings/siglip_base.npz siglip_base.npz
    modal volume put phase23-hp-tuning-data \\
      ../../../week3/phase9_polyvore_compatibility/data/positive_edges.json positive_edges.json
    modal volume put phase23-hp-tuning-data \\
      ../data/cir_val_benchmark.json cir_val_benchmark.json
    modal volume put phase23-hp-tuning-data \\
      ../../phase12_controllable_modes/data/cir_benchmark.json cir_test_benchmark.json

  Then call train_one.remote(config) or train_one.map([configs]) from an
  orchestration script's @app.local_entrypoint, or from another Modal
  function/local Python via `modal.Function.lookup`.
"""
import time

import modal

app = modal.App("phase23-hp-tuning")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
volume = modal.Volume.from_name("phase23-hp-tuning-data", create_if_missing=True)

DATA_DIR = "/data"
DEFAULT_CONFIG = {
    "name": "default",
    "lr": 1e-3,
    "batch_size": 128,
    "weight_decay": 1e-5,
    "tau": 0.07,
    "r_neg": 8,
    "seed": 42,
    "max_epochs": 12,
    "patience": 4,
    "min_delta": 0.0005,   # on validation Recall@10, not val loss
    "eval_every": 1,       # epochs between validation-benchmark Recall@K checks
    "save_checkpoint": False,
}


def _build_model_and_loss():
    import torch.nn as nn
    import torch.nn.functional as F

    class ProjectionHead(nn.Module):
        def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return F.normalize(self.net(x), p=2, dim=-1)

    def mnrl_loss(z_a, z_p, z_extra, mask, tau):
        import torch
        B = z_a.shape[0]
        inbatch_logits = (z_a @ z_p.T) / tau
        inbatch_logits = inbatch_logits.masked_fill(mask, float("-inf"))
        extra_logits = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
        logits = torch.cat([inbatch_logits, extra_logits], dim=1)
        labels = torch.arange(B, device=z_a.device)
        return F.cross_entropy(logits, labels)

    return ProjectionHead, mnrl_loss


@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=7200)
def train_one(config: dict) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    import torch

    cfg = {**DEFAULT_CONFIG, **config}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ProjectionHead, mnrl_loss = _build_model_and_loss()

    data_dir = Path(DATA_DIR)
    npz = np.load(data_dir / "siglip_base.npz", allow_pickle=True)
    item_ids = [str(a) for a in npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    embeddings = npz["embeddings"].astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    n_items = len(item_ids)
    emb_t = torch.tensor(embeddings, device=device)

    with open(data_dir / "positive_edges.json") as f:
        edge_records = json.load(f)
    train_pairs = [(e["source"], e["target"]) for e in edge_records if e["split"] == "train"]
    val_pairs = [(e["source"], e["target"]) for e in edge_records if e["split"] == "val"]

    # Integer-indexed from the start (avoids repeated string dict lookups in
    # the hot training loop -- the one real efficiency change from phase 9's
    # own script, everything else about the mechanism is identical).
    train_edges = np.array([(id_to_gidx[a], id_to_gidx[b]) for a, b in train_pairs], dtype=np.int64)
    val_edges = np.array([(id_to_gidx[a], id_to_gidx[b]) for a, b in val_pairs], dtype=np.int64)

    positive_sets = {}
    for a_idx, b_idx in np.concatenate([train_edges, val_edges], axis=0):
        positive_sets.setdefault(int(a_idx), set()).add(int(b_idx))

    with open(data_dir / "cir_val_benchmark.json") as f:
        bench = json.load(f)
    val_pools, val_queries = bench["pools"], bench["queries"]

    def evaluate_recall_gpu(model, ks=(10, 30, 50)):
        from collections import defaultdict
        model.eval()
        with torch.no_grad():
            proj = model(emb_t).cpu().numpy()
        idx = id_to_gidx
        hits = {k: 0 for k in ks}
        n_total, n_skipped = 0, 0
        by_cat = defaultdict(list)
        for qi, q in enumerate(val_queries):
            by_cat[q["category"]].append(qi)
        for cat, qidxs in by_cat.items():
            if cat not in val_pools:
                n_skipped += len(qidxs)
                continue
            pool_ids = val_pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_idx = [idx[i] for i in pool_ids]
            pool_emb = proj[pool_idx]
            qvecs, tpos = [], []
            for qi in qidxs:
                q = val_queries[qi]
                item_idx = [idx[i] for i in q["query_items"] if i in idx]
                if not item_idx or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                v = proj[item_idx].mean(axis=0)
                n = np.linalg.norm(v)
                if n == 0:
                    n_skipped += 1
                    continue
                qvecs.append(v / n)
                tpos.append(pool_pos[q["target_item"]])
            if not qvecs:
                continue
            qmat = np.stack(qvecs)
            sims = qmat @ pool_emb.T
            tpos = np.array(tpos)
            tsims = sims[np.arange(len(qvecs)), tpos]
            ranks = (sims >= tsims[:, None]).sum(axis=1)
            n_total += len(qvecs)
            for k in ks:
                hits[k] += int((ranks <= k).sum())
        recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
        return recall, n_total, n_skipped

    def sample_negatives(rng_np, anchors_idx, R):
        """Vectorized draw (no python loop over B*R in the common case):
        collisions with an anchor's own positive set or itself are
        astronomically rare (positive-set sizes are tiny relative to
        n_items=251008), so candidates are drawn in one batched call and
        only the rare colliding cells are individually resampled."""
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

    torch.manual_seed(cfg["seed"])
    rng_np = np.random.default_rng(cfg["seed"])
    model = ProjectionHead().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    B = cfg["batch_size"]
    R = cfg["r_neg"]
    tau = cfg["tau"]

    best_recall10 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    n_train = len(train_edges)
    order = np.arange(n_train)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        rng_np.shuffle(order)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, B):
            batch_idx = order[start:start + B]
            if len(batch_idx) < 2:
                continue
            batch_edges = train_edges[batch_idx]
            anchors_idx = batch_edges[:, 0]
            positives_idx = batch_edges[:, 1]
            neg_idx = sample_negatives(rng_np, anchors_idx, R)
            mask = torch.tensor(build_inbatch_mask(anchors_idx, positives_idx), device=device)

            a_emb = emb_t[torch.tensor(anchors_idx, device=device)]
            p_emb = emb_t[torch.tensor(positives_idx, device=device)]
            extra_emb = emb_t[torch.tensor(neg_idx, device=device)]

            z_a = model(a_emb)
            z_p = model(p_emb)
            Bb, K, D = extra_emb.shape
            z_extra = model(extra_emb.reshape(Bb * K, D)).reshape(Bb, K, -1)

            loss = mnrl_loss(z_a, z_p, z_extra, mask, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        do_eval = (epoch % cfg["eval_every"] == 0) or (epoch == cfg["max_epochs"] - 1)
        recall = None
        if do_eval:
            recall, n_total, n_skipped = evaluate_recall_gpu(model)
            r10 = recall[10]
            curve.append({"epoch": epoch, "train_loss": train_loss, "recall10": recall[10],
                          "recall30": recall[30], "recall50": recall[50]})
            improved = r10 > best_recall10 + cfg["min_delta"]
            if improved:
                best_recall10 = r10
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break
        else:
            curve.append({"epoch": epoch, "train_loss": train_loss})

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
        ckpt_path = data_dir / f"checkpoint_{cfg['name']}.pt"
        torch.save(best_state, ckpt_path)
        volume.commit()
        result["checkpoint_path"] = str(ckpt_path)

    return result


@app.local_entrypoint()
def smoke_test():
    """Quick manual sanity check: `modal run scripts/modal_app.py::smoke_test`
    -- trains a couple of epochs at phase 9's original hyperparameters to
    confirm the pipeline runs end to end and measure real wall-clock/epoch
    before committing to the full grid's epoch budgets."""
    import json
    cfg = {"name": "smoke_test", "max_epochs": 2, "patience": 5, "eval_every": 1}
    result = train_one.remote(cfg)
    print(json.dumps(result, indent=2))

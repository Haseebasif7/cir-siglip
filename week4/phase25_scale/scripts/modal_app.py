"""
Phase 25: Modal GPU training app for testing architecture SCALE (width,
depth, final embedding dimension) on top of phase 23/24's fully-tuned
hyperparameters. Reuses the same Modal Volume phase 23 already built and
uploaded (`phase23-hp-tuning-data` -- embeddings, positive_edges, val
benchmark, test benchmark) plus phase 25's own new train-split benchmark
(`cir_train_benchmark.json`, uploaded separately, see
00_build_train_benchmark.py) -- own app ("phase25-scale"), own function, so
this phase's runs are cleanly separated from phase 23/24's, but no data is
re-uploaded.

Per the brief: batch_size=256, weight_decay=0.0, tau=0.15, r_neg=8 are held
FIXED at their phase 23/24-tuned values throughout this entire phase --
only learning rate (rechecked at each new architecture size) and the
architecture itself (hidden_dims, out_dim) vary. Model selection and early
stopping are keyed to validation-benchmark Recall@10, never validation loss
-- same standing rule as phases 23/24.

New in this app relative to phase 23's: (1) a generalized model builder
supporting arbitrary hidden-layer widths/counts and output dimension,
instead of the fixed 768->256->128 architecture; (2) TRAIN-side Recall@10 is
also computed every eval epoch (on the subsampled train benchmark), not just
validation Recall@10 -- required for step 4's overfitting check (train vs.
val trend, not just the final validation number); (3) the collapse
diagnostic (mean pairwise cosine similarity of a random embedding sample)
is recorded every epoch for step 5's sanity check.
"""
import time

import modal

app = modal.App("phase25-scale")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
volume = modal.Volume.from_name("phase23-hp-tuning-data", create_if_missing=False)

DATA_DIR = "/data"
FIXED = {  # held fixed throughout this phase, per the brief
    "batch_size": 256,
    "weight_decay": 0.0,
    "tau": 0.15,
    "r_neg": 8,
}
DEFAULT_CONFIG = {
    "name": "default",
    "lr": 1e-3,
    "hidden_dims": [256],   # e.g. [256] = original 1-hidden-layer arch; [512,512] = wider+deeper
    "out_dim": 128,
    "seed": 42,
    "max_epochs": 12,
    "patience": 4,
    "min_delta": 0.0005,   # on validation Recall@10
    "eval_every": 1,
    "save_checkpoint": False,
    **FIXED,
}


def _build_model_and_loss():
    import torch.nn as nn
    import torch.nn.functional as F

    class ProjectionHead(nn.Module):
        """Generalized: arbitrary hidden_dims list (>=1 hidden layer),
        ReLU+Dropout(0.1) between every layer, L2-normalized output of
        dimension out_dim. hidden_dims=[256], out_dim=128 reproduces phase
        9/23/24's exact original architecture."""

        def __init__(self, in_dim=768, hidden_dims=(256,), out_dim=128, dropout=0.1):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

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


@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=21600)
def train_one(config: dict) -> dict:
    import json
    from collections import defaultdict
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

    train_edges = np.array([(id_to_gidx[a], id_to_gidx[b]) for a, b in train_pairs], dtype=np.int64)
    val_edges = np.array([(id_to_gidx[a], id_to_gidx[b]) for a, b in val_pairs], dtype=np.int64)

    positive_sets = {}
    for a_idx, b_idx in np.concatenate([train_edges, val_edges], axis=0):
        positive_sets.setdefault(int(a_idx), set()).add(int(b_idx))

    with open(data_dir / "cir_val_benchmark.json") as f:
        val_bench = json.load(f)
    val_pools, val_queries = val_bench["pools"], val_bench["queries"]

    with open(data_dir / "cir_train_benchmark.json") as f:
        train_bench = json.load(f)
    trbench_pools, trbench_queries = train_bench["pools"], train_bench["queries"]

    def evaluate_recall_gpu(proj, pools, queries, ks=(10, 30, 50)):
        idx = id_to_gidx
        hits = {k: 0 for k in ks}
        n_total, n_skipped = 0, 0
        by_cat = defaultdict(list)
        for qi, q in enumerate(queries):
            by_cat[q["category"]].append(qi)
        for cat, qidxs in by_cat.items():
            if cat not in pools:
                n_skipped += len(qidxs)
                continue
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_idx = [idx[i] for i in pool_ids]
            pool_emb = proj[pool_idx]
            qvecs, tpos = [], []
            for qi in qidxs:
                q = queries[qi]
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

    def collapse_metric(model, rng_np, n_sample=512):
        sample_idx = rng_np.choice(n_items, size=min(n_sample, n_items), replace=False)
        with torch.no_grad():
            z = model(emb_t[torch.tensor(sample_idx, device=device)])
        sims = (z @ z.T).cpu().numpy()
        n = sims.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(sims[mask].mean())

    def sample_negatives(rng_np, anchors_idx, R):
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
    model = ProjectionHead(hidden_dims=cfg["hidden_dims"], out_dim=cfg["out_dim"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
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
        if do_eval:
            model.eval()
            with torch.no_grad():
                proj = model(emb_t).cpu().numpy()
            val_recall, val_n, val_skip = evaluate_recall_gpu(proj, val_pools, val_queries)
            train_recall, train_n, train_skip = evaluate_recall_gpu(proj, trbench_pools, trbench_queries)
            collapse = collapse_metric(model, rng_np)

            r10 = val_recall[10]
            curve.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_recall10": val_recall[10], "val_recall30": val_recall[30], "val_recall50": val_recall[50],
                "train_recall10": train_recall[10], "train_recall30": train_recall[30],
                "collapse_mean_pairwise_cosine": collapse,
            })
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
        "n_params": n_params,
        "curve": curve,
        "best_epoch": best_epoch,
        "best_recall10": best_recall10,
        "n_epochs_run": len(curve),
        "wall_time_sec": wall_time,
        "device": device,
    }

    if cfg.get("save_checkpoint") and best_state is not None:
        ckpt_path = data_dir / f"phase25_checkpoint_{cfg['name']}.pt"
        torch.save(best_state, ckpt_path)
        volume.commit()
        result["checkpoint_path"] = str(ckpt_path)

    return result


@app.local_entrypoint()
def smoke_test():
    """`modal run scripts/modal_app.py::smoke_test` -- confirms the
    generalized architecture + train-benchmark eval pipeline runs end to
    end, and that hidden_dims=[256]/out_dim=128 at phase 23/24's own lr
    reproduces a comparable number to the established baseline."""
    import json
    cfg = {"name": "smoke_test", "hidden_dims": [256], "out_dim": 128, "lr": 0.001,
           "max_epochs": 2, "patience": 5, "eval_every": 1}
    result = train_one.remote(cfg)
    print(json.dumps(result, indent=2))

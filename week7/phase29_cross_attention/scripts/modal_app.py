"""
Phase 29: Modal GPU training app for candidate-conditioned cross-attention
over context, replacing phase 28's mean-pooling. Reuses phase 27/28's
already-uploaded volume ("phase27-text-category-data" -- siglip_base.npz,
text_embeddings.npz, positive_edges.json, cir_val_benchmark.json,
cir_test_benchmark.json) plus this phase's own new file
(context_training_pairs.json, uploaded by 00_build_context_training_pairs.py).
Own app ("phase29-cross-attention"), own function, so runs stay cleanly
separated from phase 27/28's.

See architecture_notes.md for the full reasoning behind the multi-item
context training data (necessary, not optional -- candidate_key/context_query
get exactly zero gradient under single-item context) and the true-positive-
conditioned training simplification (keeps training cost close to phase 28's
while still training all four new layers on genuinely multi-item context).

Modal functions must be self-contained (the function body is what actually
ships to the container), so the model classes are redefined inline here
rather than imported from model.py -- model.py is kept as the single
source of truth for local scripts (attention qualitative check) and this
file's copy is kept byte-for-byte identical to it.
"""
import time

import modal

app = modal.App("phase29-cross-attention")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
volume = modal.Volume.from_name("phase27-text-category-data", create_if_missing=False)

DATA_DIR = "/data"

DEFAULT_CONFIG = {
    "name": "default",
    "hidden_dims": [1024],
    "out_dim": 128,
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
    "eval_query_chunk": 256,  # queries per batched attention chunk at eval time (memory bound)
}


def _build_model_and_loss():
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProjectionHeadGeneral(nn.Module):
        def __init__(self, in_dim, hidden_dims=(1024,), out_dim=128, dropout=0.1):
            super().__init__()
            layers, prev = [], in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return F.normalize(self.net(x), p=2, dim=-1)

    class CrossAttentionScorer(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.dim = dim
            self.candidate_key = nn.Linear(dim, dim)
            self.context_query = nn.Linear(dim, dim)
            self.context_value = nn.Linear(dim, dim)
            self.candidate_value = nn.Linear(dim, dim)
            self.scale = math.sqrt(dim)

        def query_for_candidate(self, proj_context, proj_candidate, context_mask=None):
            k_c = self.candidate_key(proj_candidate)
            q_x = self.context_query(proj_context)
            logits = torch.einsum("...ld,...d->...l", q_x, k_c) / self.scale
            if context_mask is not None:
                logits = logits.masked_fill(context_mask, float("-inf"))
            weights = F.softmax(logits, dim=-1)
            v_x = self.context_value(proj_context)
            query = torch.einsum("...l,...ld->...d", weights, v_x)
            return F.normalize(query, p=2, dim=-1), weights

        def value_for_candidate(self, proj_candidate):
            return F.normalize(self.candidate_value(proj_candidate), p=2, dim=-1)

    def mnrl_loss(z_a, z_p, z_extra, mask, tau):
        B = z_a.shape[0]
        inbatch_logits = (z_a @ z_p.T) / tau
        inbatch_logits = inbatch_logits.masked_fill(mask, float("-inf"))
        extra_logits = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
        logits = torch.cat([inbatch_logits, extra_logits], dim=1)
        labels = torch.arange(B, device=z_a.device)
        return F.cross_entropy(logits, labels)

    return ProjectionHeadGeneral, CrossAttentionScorer, mnrl_loss


@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=7200, max_containers=5)
def train_one(config: dict) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    cfg = {**DEFAULT_CONFIG, **config}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ProjectionHeadGeneral, CrossAttentionScorer, mnrl_loss = _build_model_and_loss()
    data_dir = Path(DATA_DIR)

    # --- Credit-safety: idempotent skip + warm-start resume (unchanged
    # pattern from phase 27/28). ---
    status_path = data_dir / f"status_{cfg['name']}.json"
    ckpt_path = data_dir / f"checkpoint_{cfg['name']}.pt"
    xattn_ckpt_path = data_dir / f"checkpoint_{cfg['name']}_xattn.pt"

    if status_path.exists():
        with open(status_path) as f:
            prior_status = json.load(f)
        if prior_status.get("status") == "done":
            print(f"{cfg['name']}: already done (val_recall10="
                  f"{prior_status.get('best_recall10')}), skipping -- credit-safety short-circuit.")
            return prior_status["result"]

    # --- Load image+text base representation, identical to phase 27/28's
    # use_category="none" path. ---
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

    # --- Multi-item context training/val examples (see architecture_notes.md
    # for why this replaces the old pairwise positive_edges.json framing). ---
    with open(data_dir / "context_training_pairs.json") as f:
        context_examples = json.load(f)
    train_examples = [e for e in context_examples if e["split"] == "train"]
    val_examples = [e for e in context_examples if e["split"] == "val"]

    def to_gidx_list(ids):
        return [id_to_gidx[i] for i in ids if i in id_to_gidx]

    train_ctx = [(to_gidx_list(e["context_items"]), id_to_gidx[e["target_item"]])
                 for e in train_examples if e["target_item"] in id_to_gidx]
    train_ctx = [(c, t) for c, t in train_ctx if len(c) >= 1]

    # --- positive_sets, reused unchanged from phase 27/28 for masking and
    # negative sampling (generalized to a per-context UNION, see
    # architecture_notes.md). ---
    with open(data_dir / "positive_edges.json") as f:
        edge_records = json.load(f)
    positive_sets = {}
    for e in edge_records:
        if e["source"] in id_to_gidx and e["target"] in id_to_gidx:
            a_idx = id_to_gidx[e["source"]]
            b_idx = id_to_gidx[e["target"]]
            positive_sets.setdefault(a_idx, set()).add(b_idx)

    def forbidden_set(context_idx, target_idx):
        forb = set(context_idx)
        forb.add(target_idx)
        for c in context_idx:
            forb |= positive_sets.get(c, set())
        forb |= positive_sets.get(target_idx, set())
        return forb

    with open(data_dir / "cir_val_benchmark.json") as f:
        bench = json.load(f)
    val_pools, val_queries = bench["pools"], bench["queries"]

    model = ProjectionHeadGeneral(in_dim=1536, hidden_dims=cfg["hidden_dims"],
                                   out_dim=cfg["out_dim"]).to(device)
    xattn = CrossAttentionScorer(dim=cfg["out_dim"]).to(device)
    params = list(model.parameters()) + list(xattn.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Warm-start resume (credit-safety, unchanged pattern).
    if ckpt_path.exists():
        print(f"{cfg['name']}: found an existing checkpoint, warm-starting "
              f"(optimizer/epoch state not restored, only weights).")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if xattn_ckpt_path.exists():
            xattn.load_state_dict(torch.load(xattn_ckpt_path, map_location=device))
        if status_path.exists():
            with open(status_path) as f:
                prior_status = json.load(f)
            best_recall10 = prior_status.get("best_recall10", -1.0) or -1.0
            print(f"  warm-started at best_recall10={best_recall10}")
    else:
        best_recall10 = -1.0

    def pad_context(context_idx_batch, device):
        """List of variable-length int lists -> (B, Lmax) padded index tensor
        + (B, Lmax) bool mask (True = padding)."""
        Lmax = max(len(c) for c in context_idx_batch)
        B = len(context_idx_batch)
        idx_t = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        mask_t = torch.ones((B, Lmax), dtype=torch.bool, device=device)
        for i, c in enumerate(context_idx_batch):
            idx_t[i, :len(c)] = torch.tensor(c, device=device)
            mask_t[i, :len(c)] = False
        return idx_t, mask_t

    def evaluate_recall_gpu(model, xattn, ks=(10, 30, 50), chunk=256):
        from collections import defaultdict
        model.eval()
        xattn.eval()
        with torch.no_grad():
            proj_all = model(base_repr)  # (n_items, out_dim)
            cand_key_all = xattn.candidate_key(proj_all)
            cand_val_all = F.normalize(xattn.candidate_value(proj_all), p=2, dim=-1)
            ctx_query_all = xattn.context_query(proj_all)
            ctx_value_all = xattn.context_value(proj_all)

            by_cat = defaultdict(list)
            for qi, q in enumerate(val_queries):
                by_cat[q["category"]].append(qi)

            hits = {k: 0 for k in ks}
            n_total, n_skipped = 0, 0
            scale = float(xattn.scale)

            for cat, qidxs in by_cat.items():
                if cat not in val_pools:
                    n_skipped += len(qidxs)
                    continue
                pool_ids = val_pools[cat]
                pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
                pool_idx = [id_to_gidx[i] for i in pool_ids if i in id_to_gidx]
                pool_key = cand_key_all[pool_idx]     # (P, d)
                pool_val = cand_val_all[pool_idx]      # (P, d)

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

                tpos = np.array(tpos)
                all_ranks = np.zeros(len(kept_qidx), dtype=np.int64)

                # Batched, candidate-conditioned attention per chunk of
                # queries -- true per-(query,candidate) scoring, not an
                # approximation, chunked only to bound peak memory.
                for start in range(0, len(kept_ctx_idx), chunk):
                    sub_ctx = kept_ctx_idx[start:start + chunk]
                    ctx_idx_t, ctx_mask_t = pad_context(sub_ctx, device)  # (b, Lmax)
                    q_x = ctx_query_all[ctx_idx_t]                          # (b, Lmax, d)
                    v_x = ctx_value_all[ctx_idx_t]                          # (b, Lmax, d)
                    # logits[b, p, l] = q_x[b,l,:] . pool_key[p,:] / scale
                    logits = torch.einsum("bld,pd->bpl", q_x, pool_key) / scale
                    pad_mask = ctx_mask_t.unsqueeze(1)  # (b, 1, Lmax) broadcasts over P
                    logits = logits.masked_fill(pad_mask, float("-inf"))
                    weights = F.softmax(logits, dim=-1)                     # (b, P, Lmax)
                    query_per_cand = torch.einsum("bpl,bld->bpd", weights, v_x)  # (b, P, d)
                    query_per_cand = F.normalize(query_per_cand, p=2, dim=-1)
                    scores = (query_per_cand * pool_val.unsqueeze(0)).sum(-1)  # (b, P)
                    scores_np = scores.cpu().numpy()
                    sub_tpos = tpos[start:start + chunk]
                    tsims = scores_np[np.arange(len(sub_ctx)), sub_tpos]
                    ranks = (scores_np >= tsims[:, None]).sum(axis=1)
                    all_ranks[start:start + chunk] = ranks

                n_total += len(kept_qidx)
                for k in ks:
                    hits[k] += int((all_ranks <= k).sum())

        recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
        return recall, n_total, n_skipped

    rng_np = np.random.default_rng(cfg["seed"])

    def sample_negatives_for_example(context_idx, target_idx, R):
        forb = forbidden_set(context_idx, target_idx)
        neg = []
        tries = 0
        while len(neg) < R:
            c = int(rng_np.integers(0, n_items))
            tries += 1
            if c not in forb and c not in neg:
                neg.append(c)
            if tries > 40 * R:  # pathological fallback, never expected to trigger
                while len(neg) < R:
                    neg.append(int(rng_np.integers(0, n_items)))
        return neg

    order = np.arange(len(train_ctx))
    B = cfg["batch_size"]
    R = cfg["r_neg"]
    tau = cfg["tau"]

    best_state = None
    best_xattn_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    for epoch in range(cfg["max_epochs"]):
        model.train()
        xattn.train()
        rng_np.shuffle(order)
        total_loss, n_batches = 0.0, 0
        entropy_sum, entropy_norm_sum, n_entropy = 0.0, 0.0, 0

        for start in range(0, len(train_ctx), B):
            batch_idx = order[start:start + B]
            if len(batch_idx) < 2:
                continue
            batch = [train_ctx[i] for i in batch_idx]
            context_idx_batch = [c for c, t in batch]
            target_idx_batch = [t for c, t in batch]

            ctx_idx_t, ctx_mask_t = pad_context(context_idx_batch, device)  # (B, Lmax)
            target_t = torch.tensor(target_idx_batch, device=device)

            neg_idx = [sample_negatives_for_example(c, t, R) for c, t in batch]
            neg_t = torch.tensor(neg_idx, device=device)  # (B, R)

            # False-negative in-batch mask, generalized to the per-example
            # forbidden set (context union + target's own positive edges).
            mask = np.zeros((len(batch), len(batch)), dtype=bool)
            forb_sets = [forbidden_set(c, t) for c, t in batch]
            for i in range(len(batch)):
                for j in range(len(batch)):
                    if i != j and target_idx_batch[j] in forb_sets[i]:
                        mask[i, j] = True
            mask_t = torch.tensor(mask, device=device)

            proj_context = model(base_repr[ctx_idx_t])            # (B, Lmax, out_dim)
            proj_target = model(base_repr[target_t])              # (B, out_dim)
            proj_neg = model(base_repr[neg_t].reshape(-1, 1536)).reshape(len(batch), R, -1)

            z_a, attn_w = xattn.query_for_candidate(proj_context, proj_target, context_mask=ctx_mask_t)
            z_p = xattn.value_for_candidate(proj_target)
            z_extra = xattn.value_for_candidate(proj_neg.reshape(-1, proj_neg.shape[-1])).reshape(
                len(batch), R, -1)

            loss = mnrl_loss(z_a, z_p, z_extra, mask_t, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            # Attention entropy diagnostic (collapse check), computed on this
            # training batch's own weights -- cheap, no extra forward pass.
            with torch.no_grad():
                p = attn_w.clamp_min(1e-12)
                ent = -(p * p.log()).sum(dim=-1)  # (B,)
                ctx_lens = (~ctx_mask_t).sum(dim=-1).clamp_min(2).float()  # avoid log(1)=0 div
                ent_norm = ent / ctx_lens.log()
                valid = (~ctx_mask_t).sum(dim=-1) > 1  # entropy undefined/trivial for len-1 context
                if valid.any():
                    entropy_sum += ent[valid].sum().item()
                    entropy_norm_sum += ent_norm[valid].sum().item()
                    n_entropy += int(valid.sum().item())

        train_loss = total_loss / max(n_batches, 1)
        mean_entropy = entropy_sum / max(n_entropy, 1)
        mean_entropy_norm = entropy_norm_sum / max(n_entropy, 1)

        do_eval = (epoch % cfg["eval_every"] == 0) or (epoch == cfg["max_epochs"] - 1)
        if do_eval:
            recall, n_total, n_skipped = evaluate_recall_gpu(model, xattn, chunk=cfg["eval_query_chunk"])
            r10 = recall[10]
            curve.append({"epoch": epoch, "train_loss": train_loss, "recall10": recall[10],
                          "recall30": recall[30], "recall50": recall[50],
                          "mean_attn_entropy": mean_entropy, "mean_attn_entropy_norm": mean_entropy_norm})
            improved = r10 > best_recall10 + cfg["min_delta"]
            if improved:
                best_recall10 = r10
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_xattn_state = {k: v.clone().cpu() for k, v in xattn.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0

                if cfg.get("save_checkpoint"):
                    torch.save(best_state, ckpt_path)
                    torch.save(best_xattn_state, xattn_ckpt_path)
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
                          "mean_attn_entropy": mean_entropy, "mean_attn_entropy_norm": mean_entropy_norm})

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
        torch.save(best_xattn_state, xattn_ckpt_path)
        result["checkpoint_path"] = str(ckpt_path)
        result["xattn_checkpoint_path"] = str(xattn_ckpt_path)
        with open(status_path, "w") as f:
            json.dump({"status": "done", "best_epoch": best_epoch,
                        "best_recall10": best_recall10, "result": result}, f)
        volume.commit()

    return result


@app.local_entrypoint()
def smoke_test():
    """`modal run scripts/modal_app.py::smoke_test` -- 2-epoch sanity check."""
    import json
    cfg = {"name": "smoke_xattn", "max_epochs": 2, "patience": 5, "eval_every": 1}
    result = train_one.remote(cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    for row in result["curve"]:
        print(row)

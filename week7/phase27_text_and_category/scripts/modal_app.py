"""
Phase 27: Modal GPU training app for the two required variants (text only;
text + category conditioning) plus the category-encoding-method secondary
check (learned table vs. SigLIP-encoded category phrase, both under
"text + category"). Reuses phase 23's already-deployed data volume
("phase23-hp-tuning-data" -- siglip_base.npz, positive_edges.json,
cir_val_benchmark.json, cir_test_benchmark.json) rather than re-uploading
any of it; this phase's own new files (text_embeddings.npz,
category_index.npz, category_text_embeddings.npz) are added to that same
volume. Own app ("phase27-text-category"), own function, so runs stay
cleanly separated from phase 23/25/26's.

Architecture and mechanism (see implementation_notes.md for the full
reasoning behind each decision):

- Per-item base representation: base_repr(i) = normalize(concat(image(i),
  text(i))) -- 1536-d, computed once for the whole catalog, used for every
  item regardless of role (context item, candidate, or training
  anchor/positive/negative).
- use_category="none" (variant 1, "text only"): candidate_repr = base_repr
  (1536-d). Every item -- training anchor/positive/negative AND every eval
  candidate/context item -- is projected individually through ONE shared
  head. Query construction at eval time is IDENTICAL to phase 23/25/26's own
  mechanism: project the whole catalog once, then mean-pool the ALREADY-
  PROJECTED (128-d) context-item embeddings per query. This keeps variant 1
  a clean, minimal, apples-to-apples widening of week 6's winner (only
  difference: text is now part of each item's input) -- directly comparable
  to week 6's reported number, nothing else about the mechanism changes.
- use_category in {"learned","siglip_phrase"} (variants 2/3, decision #4):
  candidate_repr = pad(base_repr, 768 zeros) -- 2304-d, same shared head,
  same "project individually" treatment for every candidate/context item
  (the zero category slot contributes nothing beyond the head's own bias,
  consistent with the brief's zero-vector-for-missing convention). The QUERY
  is different by necessity: decision #4 requires category to be
  concatenated onto the query vector BEFORE the (single, shared) projection
  head runs, so pooling for the query happens on RAW base_repr vectors
  (pre-projection), category is concatenated, the whole 2304-d vector is
  renormalized, and ONE head call produces the query embedding. This is a
  deliberate, necessary asymmetry against variant 1's post-projection
  pooling -- flagged explicitly in implementation_notes.md, not hidden.
  During TRAINING, the "anchor" role plays the part of an eval-time query
  with context size exactly 1 -- pooling over one item is that item, so this
  asymmetry never actually bites during training itself, only at eval time
  when real queries have >1 context item. The anchor's category is the
  POSITIVE/target item's own semantic_category (what the query is trying to
  retrieve) -- exactly mirroring the CIR benchmark's own query["category"]
  field used at eval time.
"""
import time

import modal

app = modal.App("phase27-text-category")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
# Own volume under the m-haseebasif5 workspace (switched from muhamasif
# mid-phase -- see implementation_notes.md's "Modal account switch" section).
# Holds phase 23's reused data (siglip_base.npz, positive_edges.json,
# cir_val_benchmark.json, cir_test_benchmark.json, re-uploaded fresh to this
# workspace) plus this phase's own new files (text_embeddings.npz,
# category_index.npz, category_text_embeddings.npz).
volume = modal.Volume.from_name("phase27-text-category-data", create_if_missing=False)

DATA_DIR = "/data"
CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]

# Week 6's tuned hyperparameters, held fixed per the brief -- only the input
# construction (use_category) varies across this phase's runs.
DEFAULT_CONFIG = {
    "name": "default",
    "use_category": "none",       # "none" | "learned" | "siglip_phrase"
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
}


def _build_model_and_loss():
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

    def mnrl_loss(z_a, z_p, z_extra, mask, tau):
        import torch
        B = z_a.shape[0]
        inbatch_logits = (z_a @ z_p.T) / tau
        inbatch_logits = inbatch_logits.masked_fill(mask, float("-inf"))
        extra_logits = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
        logits = torch.cat([inbatch_logits, extra_logits], dim=1)
        labels = torch.arange(B, device=z_a.device)
        return F.cross_entropy(logits, labels)

    return ProjectionHeadGeneral, mnrl_loss


# max_containers caps how many T4 containers this function will scale to at
# once. Added after the m-haseebasif5 workspace's GPU concurrency quota was
# hit launching phase 28's 9-seed .map() (~9-10 concurrent containers) --
# throttles future .map() calls to queue past this cap instead of over-
# requesting and hitting the account limit again. Does not affect containers
# already running under the previous deployment.
@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=7200, max_containers=5)
def train_one(config: dict) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = {**DEFAULT_CONFIG, **config}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ProjectionHeadGeneral, mnrl_loss = _build_model_and_loss()
    data_dir = Path(DATA_DIR)

    # --- Credit-safety: idempotent skip + warm-start resume. Added after
    # switching Modal accounts mid-phase (muhamasif -> m-haseebasif5, fresh
    # workspace, unknown credit budget) -- see implementation_notes.md.
    # A run already marked "done" on the volume is returned immediately
    # without spending any more compute; a run that got INTERRUPTED partway
    # (checkpoint exists, status != "done") warm-starts from that checkpoint
    # rather than from a random init, so an interrupted run's partial
    # progress is never silently thrown away. This is a warm start, not a
    # byte-exact resume (optimizer state, epoch counter, and patience
    # counter are NOT restored) -- simpler to implement correctly, and the
    # improvement-only checkpoint logic below means further training can
    # only match or improve on the warm-started weights, never regress.
    status_path = data_dir / f"status_{cfg['name']}.json"
    ckpt_path = data_dir / f"checkpoint_{cfg['name']}.pt"
    cat_ckpt_path = data_dir / f"checkpoint_{cfg['name']}_category_table.pt"

    if status_path.exists():
        with open(status_path) as f:
            prior_status = json.load(f)
        if prior_status.get("status") == "done":
            print(f"{cfg['name']}: already done (val_recall10="
                  f"{prior_status.get('best_recall10')}), skipping -- credit-safety short-circuit.")
            return prior_status["result"]

    # --- Load and align image, text, category data (all positionally
    # aligned to siglip_base.npz's item_ids order by construction). ---
    img_npz = np.load(data_dir / "siglip_base.npz", allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    n_items = len(item_ids)

    image_emb = img_npz["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_npz = np.load(data_dir / "text_embeddings.npz", allow_pickle=True)
    text_ids = [str(a) for a in text_npz["item_ids"]]
    assert text_ids == item_ids, "text_embeddings.npz must be positionally aligned to siglip_base.npz"
    text_emb = text_npz["embeddings"].astype(np.float32)  # already L2-normalized at extraction time,
    # renormalized defensively below anyway since concat+renorm is the real invariant that matters.

    cat_npz = np.load(data_dir / "category_index.npz", allow_pickle=True)
    cat_ids = [str(a) for a in cat_npz["item_ids"]]
    assert cat_ids == item_ids, "category_index.npz must be positionally aligned to siglip_base.npz"
    category_idx_np = cat_npz["category_idx"].astype(np.int64)  # -1 for unknown (none expected)

    use_category = cfg["use_category"]
    assert use_category in ("none", "learned", "siglip_phrase")

    torch.manual_seed(cfg["seed"])

    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)  # (n_items, 1536)
    category_idx_t = torch.tensor(category_idx_np, device=device)

    if use_category == "none":
        candidate_repr = base_repr  # (n_items, 1536)
        in_dim = 1536
        category_table = None
        category_lookup = None
    else:
        candidate_repr = F.pad(base_repr, (0, 768))  # (n_items, 2304), zero category slot
        in_dim = 2304
        if use_category == "learned":
            category_table = nn.Embedding(len(CATEGORY_LIST), 768).to(device)
            nn.init.normal_(category_table.weight, std=0.02)
            category_lookup = None
        else:  # siglip_phrase -- fixed, not trained
            cat_text_npz = np.load(data_dir / "category_text_embeddings.npz", allow_pickle=True)
            cat_text_cats = [str(c) for c in cat_text_npz["categories"]]
            assert cat_text_cats == CATEGORY_LIST, "category_text_embeddings.npz category order mismatch"
            cat_vecs = cat_text_npz["embeddings"].astype(np.float32)
            cat_vecs = cat_vecs / np.linalg.norm(cat_vecs, axis=1, keepdims=True)
            category_lookup = torch.tensor(cat_vecs, device=device)  # (11, 768), constant
            category_table = None

    def category_vecs_for(cat_idx_batch):
        """cat_idx_batch: (B,) long tensor of category indices -> (B, 768)
        L2-normalized category vectors, from whichever source this run uses."""
        if category_table is not None:
            v = category_table(cat_idx_batch)
        else:
            v = category_lookup[cat_idx_batch]
        return F.normalize(v, p=2, dim=-1)

    def build_anchor_repr(anchor_idx_t, target_cat_idx_t):
        if use_category == "none":
            return candidate_repr[anchor_idx_t]
        a_base = base_repr[anchor_idx_t]                    # (B, 1536)
        cat_vecs = category_vecs_for(target_cat_idx_t)       # (B, 768)
        return F.normalize(torch.cat([a_base, cat_vecs], dim=-1), p=2, dim=-1)

    # --- Training/validation edges (single-item anchor -> positive pairs,
    # identical data source and split to phase 9/23/25/26). ---
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
    CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}

    def evaluate_recall_gpu(model, ks=(10, 30, 50)):
        from collections import defaultdict
        model.eval()
        with torch.no_grad():
            proj_candidates = model(candidate_repr).cpu().numpy()  # (n_items, out_dim)

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
                pool_emb = proj_candidates[pool_idx]

                # Build each kept query's context-item index list once, shared
                # by both the "none" and category-conditioned paths below.
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

                if use_category == "none":
                    # Post-projection pooling -- identical to phase 23/25/26.
                    qvecs = []
                    for ctx_idx in kept_ctx_idx:
                        v = proj_candidates[ctx_idx].mean(axis=0)
                        n = np.linalg.norm(v)
                        qvecs.append(v / n if n > 0 else v)
                    qmat = np.stack(qvecs)
                else:
                    # Pre-projection pooling + category concat, per decision #4:
                    # one batched forward call for every query in this category.
                    target_cat_idx = CATEGORY_TO_IDX.get(cat)
                    pooled_raws = []
                    with torch.no_grad():
                        for ctx_idx in kept_ctx_idx:
                            v = base_repr[torch.tensor(ctx_idx, device=device)].mean(dim=0)
                            v = F.normalize(v, p=2, dim=-1)
                            pooled_raws.append(v)
                        pooled_raw_t = torch.stack(pooled_raws)  # (nq, 1536)
                        cat_idx_t = torch.full((len(kept_qidx),), target_cat_idx,
                                                dtype=torch.long, device=device)
                        cat_vecs_t = category_vecs_for(cat_idx_t)  # (nq, 768)
                        q_repr = F.normalize(torch.cat([pooled_raw_t, cat_vecs_t], dim=-1), p=2, dim=-1)
                        qmat = model(q_repr).cpu().numpy()

                tpos = np.array(tpos)
                sims = qmat @ pool_emb.T
                tsims = sims[np.arange(len(kept_qidx)), tpos]
                ranks = (sims >= tsims[:, None]).sum(axis=1)
                n_total += len(kept_qidx)
                for k in ks:
                    hits[k] += int((ranks <= k).sum())

        recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
        return recall, n_total, n_skipped

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

    model = ProjectionHeadGeneral(in_dim=in_dim, hidden_dims=cfg["hidden_dims"],
                                   out_dim=cfg["out_dim"]).to(device)
    params = list(model.parameters())
    if category_table is not None:
        params += list(category_table.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    B = cfg["batch_size"]
    R = cfg["r_neg"]
    tau = cfg["tau"]

    best_recall10 = -1.0
    best_state = None
    best_cat_state = None
    best_epoch = -1
    patience_counter = 0
    curve = []
    t_start = time.time()

    # Warm-start resume (credit-safety): a checkpoint left by an interrupted
    # earlier attempt at this same run name is loaded as the starting point
    # instead of a random init. Not a byte-exact resume (see the comment at
    # the top of this function) -- just ensures an interruption never throws
    # away all prior progress on this specific variant.
    if ckpt_path.exists():
        print(f"{cfg['name']}: found an existing checkpoint, warm-starting from it "
              f"(optimizer/epoch state is NOT restored, only weights).")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if category_table is not None and cat_ckpt_path.exists():
            category_table.load_state_dict(torch.load(cat_ckpt_path, map_location=device))
        if status_path.exists():
            with open(status_path) as f:
                prior_status = json.load(f)
            best_recall10 = prior_status.get("best_recall10", -1.0) or -1.0
            print(f"  warm-started at best_recall10={best_recall10}")

    n_train = len(train_edges)
    order = np.arange(n_train)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        if category_table is not None:
            category_table.train()
        rng_np.shuffle(order)
        total_loss, n_batches = 0.0, 0
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

            target_cat_t = category_idx_t[positives_t] if use_category != "none" else None
            a_repr = build_anchor_repr(anchors_t, target_cat_t)
            p_repr = candidate_repr[positives_t]
            extra_repr = candidate_repr[neg_t]  # (B, R, in_dim)

            z_a = model(a_repr)
            z_p = model(p_repr)
            Bb, Rr, D = extra_repr.shape
            z_extra = model(extra_repr.reshape(Bb * Rr, D)).reshape(Bb, Rr, -1)

            loss = mnrl_loss(z_a, z_p, z_extra, mask, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        do_eval = (epoch % cfg["eval_every"] == 0) or (epoch == cfg["max_epochs"] - 1)
        if do_eval:
            recall, n_total, n_skipped = evaluate_recall_gpu(model)
            r10 = recall[10]
            curve.append({"epoch": epoch, "train_loss": train_loss, "recall10": recall[10],
                          "recall30": recall[30], "recall50": recall[50]})
            improved = r10 > best_recall10 + cfg["min_delta"]
            if improved:
                best_recall10 = r10
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                if category_table is not None:
                    best_cat_state = {k: v.clone().cpu() for k, v in category_table.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0

                # Credit-safety: persist the moment there's an improvement,
                # not just once at the very end -- if this container gets
                # killed by a credit limit mid-run, the best-so-far
                # checkpoint has already reached the volume.
                if cfg.get("save_checkpoint"):
                    torch.save(best_state, ckpt_path)
                    if best_cat_state is not None:
                        torch.save(best_cat_state, cat_ckpt_path)
                    with open(status_path, "w") as f:
                        json.dump({"status": "in_progress", "best_epoch": best_epoch,
                                    "best_recall10": best_recall10, "epoch_reached": epoch}, f)
                    volume.commit()
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
        "in_dim": in_dim,
    }

    if cfg.get("save_checkpoint") and best_state is not None:
        torch.save(best_state, ckpt_path)
        result["checkpoint_path"] = str(ckpt_path)
        if best_cat_state is not None:
            torch.save(best_cat_state, cat_ckpt_path)
            result["category_checkpoint_path"] = str(cat_ckpt_path)
        with open(status_path, "w") as f:
            json.dump({"status": "done", "best_epoch": best_epoch,
                        "best_recall10": best_recall10, "result": result}, f)
        volume.commit()

    return result


@app.local_entrypoint()
def smoke_test():
    """`modal run scripts/modal_app.py::smoke_test` -- 2-epoch sanity check
    for all three use_category modes before committing to full runs."""
    import json
    for use_category in ("none", "learned", "siglip_phrase"):
        cfg = {"name": f"smoke_{use_category}", "use_category": use_category,
               "max_epochs": 2, "patience": 5, "eval_every": 1}
        result = train_one.remote(cfg)
        print(use_category, json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))

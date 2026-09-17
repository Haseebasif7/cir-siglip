"""
Phase 36 shared code: data loading, model loading, and three system adapters
(ours / OutfitTransformer / CSA-Net) whose CIR scoring paths are faithful copies
of each phase's own final-evaluation code, with one addition -- per-query ranks
are RETURNED instead of being collapsed into aggregate counters and discarded.

Nothing here trains or selects anything. The reproduction guard in
01_capture_per_query_ranks.py is what licenses trusting these copies: every
aggregate Recall@K they produce must match the number each phase already
reported.

Why model classes are loaded via importlib under distinct names: every phase's
architecture file is called `model.py`, and a bare `import model` binds whichever
one is first on sys.path (phase 17 hit exactly this; its `_load_module` pattern is
reused here).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede `import torch`

import importlib.util
import json
import resource
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

# ---------------------------------------------------------------- paths
SCRIPTS_DIR = Path(__file__).resolve().parent
PHASE_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE_DIR / "data"
FIG_DIR = PHASE_DIR / "figures"
LOG_DIR = PHASE_DIR / "logs"
REPO_ROOT = PHASE_DIR.parents[1]

PHASE9_DIR = REPO_ROOT / "week3" / "phase9_polyvore_compatibility"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
METADATA_JSON = RAW_DIR / "polyvore_item_metadata.json"
TEXT_EMBEDDINGS_NPZ = REPO_ROOT / "week7" / "phase27_text_and_category" / "data" / "text_embeddings.npz"
TEST_BENCHMARK = REPO_ROOT / "week4" / "phase12_controllable_modes" / "data" / "cir_benchmark.json"
TRAINING_DATA = REPO_ROOT / "week4" / "phase13_csa_net_baseline" / "data" / "training_data.json"

PHASE27_MODELS = REPO_ROOT / "week7" / "phase27_text_and_category" / "models"
PHASE28_MODELS = REPO_ROOT / "week7" / "phase28_text_ensemble" / "models"
PHASE32_DIR = REPO_ROOT / "week7" / "phase32_partial_ensemble_outfittransformer"
PHASE34_DIR = REPO_ROOT / "week7" / "phase34_csanet_random_negatives"
PHASE9_MODEL_A = PHASE9_DIR / "models" / "model_a_random_negs.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KS = (10, 30, 50)

# Previously reported TEST numbers -- the reproduction guard. 4-decimal values are
# what the phase markdown files report; exact floats where a JSON exists.
RECORDED_4DP = {
    "ours_ens": {10: 0.1904, 30: 0.3267, 50: 0.4079},
    "ours_solo": {10: 0.1656, 30: 0.2947, 50: 0.3733},
    "ot_ens": {10: 0.1897, 30: 0.3246, 50: 0.4019},
    "ot_solo": {10: 0.1799, 30: 0.3111, 50: 0.3844},
    "csa_ens": {10: 0.1674, 30: 0.2860, 50: 0.3586},
}
RECORDED_EXACT = {
    "ot_ens": {10: 0.18971732758330245, 30: 0.3245847511876285, 50: 0.40187325224891346},
    "ot_solo": {10: 0.17987938411778578, 30: 0.3110744247161484, 50: 0.3843873184865739},
    "csa_ens": {10: 0.1673798052626259, 30: 0.2860078838314073, 50: 0.358579562683198},
}
N_PARAMS = {"ours": 1_705_088, "ot": 998_144, "csa": 100_485}
SYSTEM_LABELS = {
    "ours_ens": "Ours, mean-pooled projection, 10-seed ensemble (phase 28)",
    "ours_solo": "Ours, mean-pooled projection, single model (phase 27 text_only)",
    "ot_ens": "OutfitTransformer mechanism, frozen backbone, matched optimization, 3-seed ensemble (phase 32)",
    "ot_solo": "OutfitTransformer mechanism, frozen backbone, matched optimization, single model (phase 31/32 seed 42)",
    "csa_ens": "CSA-Net mechanism, frozen backbone, matched optimization + random negatives, 3-seed ensemble (phase 34)",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE_CACHE = {}


def phase_module(name, path):
    if name not in _MODULE_CACHE:
        _MODULE_CACHE[name] = _load_module(name, path)
    return _MODULE_CACHE[name]


def peak_rss_mb():
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def mps_allocated_mb():
    if DEVICE == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------- data
def load_item_data(device=DEVICE):
    """Identical construction to phase 28/32/34: image block re-L2-normalized in
    NumPy, text block used as stored, concat to 1536-d, single L2 over the whole
    vector. Returns the raw image embeddings too (needed for the untrained
    raw-SigLIP guard under the official protocol)."""
    img = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img["item_ids"]]
    idx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    txt = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in txt["item_ids"]] == item_ids, "text embeddings must align to image embeddings"
    text_emb = txt["embeddings"].astype(np.float32)
    image_t = torch.tensor(image_emb, device=device)
    text_t = torch.tensor(text_emb, device=device)
    base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)
    del text_t
    return item_ids, idx, image_emb, base_repr


def load_benchmark(path=TEST_BENCHMARK):
    with open(path) as f:
        data = json.load(f)
    return data["pools"], data["queries"]


def load_category_data():
    """Re-implements the category part of phase 34 train_core.load_catalog in
    the same order (training_data first, metadata fallback second) -- not
    imported because train_core does a bare `from model import ...`."""
    with open(TRAINING_DATA) as f:
        td = json.load(f)
    categories = td["categories"]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    item_cat = {}
    for cat, items in td["train_items_by_category"].items():
        for i in items:
            item_cat[i] = cat
    for cat, items in td["val_items_by_category"].items():
        for i in items:
            item_cat[i] = cat
    with open(METADATA_JSON) as f:
        meta = json.load(f)
    for item_id, v in meta.items():
        cat = v.get("semantic_category")
        if item_id not in item_cat and cat in categories:
            item_cat[item_id] = cat
    return categories, cat_to_idx, item_cat


# ---------------------------------------------------------------- model classes
class ProjectionHeadGeneral(nn.Module):
    """Verbatim from week7/phase28_text_ensemble/scripts/cir_eval_ensemble.py."""

    def __init__(self, in_dim=1536, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ProjectionHeadPhase9(nn.Module):
    """Verbatim from week3/phase9_polyvore_compatibility/scripts/model.py."""

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


OT_MODEL_KWARGS = dict(siglip_dim=1536, d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512, dropout=0.1)
OT_QUERY_BATCH = 1024
CSA_QUERY_CHUNK = 512


def load_ot_member(ckpt_path, device=DEVICE):
    mod = phase_module("phase32_model", PHASE32_DIR / "scripts" / "model.py")
    model = mod.OutfitTransformerSigLIP(**OT_MODEL_KWARGS).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def load_csa_member(ckpt_path, num_categories, device=DEVICE):
    mod = phase_module("phase34_model", PHASE34_DIR / "scripts" / "model.py")
    model = mod.CSANetSigLIP(num_categories=num_categories, siglip_dim=1536).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _pad_gidx(gidx_lists):
    """Right-pad variable-length gidx lists with 0. Returns (B, Lmax) int64 and a
    bool VALID mask (True = real item)."""
    B = len(gidx_lists)
    Lmax = max(len(g) for g in gidx_lists)
    arr = np.zeros((B, Lmax), dtype=np.int64)
    valid = np.zeros((B, Lmax), dtype=bool)
    for i, g in enumerate(gidx_lists):
        arr[i, :len(g)] = g
        valid[i, :len(g)] = True
    return arr, valid


# ================================================================ OURS
class OursSystem:
    """Phase 28's mechanism: project every catalog item once per member (single
    whole-catalog call, as the original did), mean-pool the PROJECTED context
    vectors, renormalize, cosine to projected candidates, average cosines across
    members. Scoring in NumPy on CPU, as the original."""
    kind = "ours"
    higher_is_better = True

    def __init__(self, name, ckpt_paths, device=DEVICE):
        self.name, self.ckpt_paths, self.device = name, list(ckpt_paths), device
        self.proj_list = None
        self.n_params_per_member = None

    def precompute(self, base_repr):
        t0 = time.perf_counter()
        self.proj_list = []
        for ckpt in self.ckpt_paths:
            model = ProjectionHeadGeneral(in_dim=base_repr.shape[1], hidden_dims=(1024,), out_dim=128).to(self.device).eval()
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.n_params_per_member = count_params(model)
            with torch.no_grad():
                self.proj_list.append(model(base_repr).cpu().numpy())
            del model
        return time.perf_counter() - t0

    @staticmethod
    def _query_vector(embeddings, idx, query_items):
        # verbatim phase 28
        item_idx = [idx[i] for i in query_items if i in idx]
        if not item_idx:
            return None
        v = embeddings[item_idx].mean(axis=0)
        norm = np.linalg.norm(v)
        if norm == 0:
            return None
        return v / norm

    def cir_ranks(self, pools, queries, idx, item_ids=None):
        """Faithful copy of phase 28 evaluate_recall_ensemble; the only change is
        that (qidx_kept, ranks) are recorded instead of being summed and dropped."""
        t0 = time.perf_counter()
        embeddings_list = self.proj_list
        ranks_out = np.full(len(queries), -1, dtype=np.int32)
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
            qidx_kept, target_positions = [], []
            for qi in qidxs:
                q = queries[qi]
                qv0 = self._query_vector(embeddings_list[0], idx, q["query_items"])
                if qv0 is None or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                qidx_kept.append(qi)
                target_positions.append(pool_pos[q["target_item"]])
            if not qidx_kept:
                continue
            sims_sum = None
            for emb in embeddings_list:
                pool_emb = emb[pool_idx]
                qvecs = [self._query_vector(emb, idx, queries[qi]["query_items"]) for qi in qidx_kept]
                query_mat = np.stack(qvecs)
                sims = query_mat @ pool_emb.T
                sims_sum = sims if sims_sum is None else sims_sum + sims
            sims_avg = sims_sum / len(embeddings_list)
            target_positions = np.array(target_positions)
            target_sims = sims_avg[np.arange(len(qidx_kept)), target_positions]
            ranks = (sims_avg >= target_sims[:, None]).sum(axis=1)
            ranks_out[np.array(qidx_kept)] = ranks
            n_total += len(qidx_kept)
        return ranks_out, n_total, n_skipped, time.perf_counter() - t0

    def score_sets(self, ctx_gidx_lists, cand_gidx_lists, ctx_cat_lists=None, cand_cat_lists=None, base_repr=None, chunk=4096):
        """Generic scorer for the official protocol: scores[b, a] = ensemble-average
        cosine between the normalized mean of projected context b and projected
        candidate a. Vectorized equivalent of _query_vector over padded contexts."""
        B = len(ctx_gidx_lists)
        A = len(cand_gidx_lists[0])
        out = np.zeros((B, A), dtype=np.float64)
        for start in range(0, B, chunk):
            ctx_arr, valid = _pad_gidx(ctx_gidx_lists[start:start + chunk])
            cand_arr = np.asarray(cand_gidx_lists[start:start + chunk], dtype=np.int64)
            n_ctx = valid.sum(axis=1, keepdims=True).astype(np.float32)
            acc = None
            for emb in self.proj_list:
                ctx_e = emb[ctx_arr] * valid[:, :, None]
                q = ctx_e.sum(axis=1) / n_ctx
                q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
                c = emb[cand_arr]  # (b, A, D)
                s = np.einsum("bd,bad->ba", q, c)
                acc = s if acc is None else acc + s
            out[start:start + chunk] = acc / len(self.proj_list)
        return out

    def item_embeddings(self):
        """Per-member unconditioned item embedding matrices (for the secondary,
        phase-9-style protocol)."""
        return self.proj_list


# ================================================================ OUTFITTRANSFORMER
class OTSystem:
    """Phase 32's mechanism. Candidates: embed_item_alone over the whole catalog per
    member (chunks of 4096). Queries: outfit token + context through the set
    encoder (QUERY_BATCH=1024). Cosine, averaged across members."""
    kind = "ot"
    higher_is_better = True

    def __init__(self, name, ckpt_paths, device=DEVICE):
        self.name, self.ckpt_paths, self.device = name, list(ckpt_paths), device
        self.members = None
        self.n_params_per_member = None

    def precompute(self, base_repr):
        t0 = time.perf_counter()
        self.members = []
        for ckpt in self.ckpt_paths:
            model = load_ot_member(ckpt, self.device)
            self.n_params_per_member = count_params(model)
            outs = []
            with torch.no_grad():
                for start in range(0, base_repr.shape[0], 4096):
                    chunk = base_repr[start:start + 4096]
                    tokens = model.encode_item_tokens(chunk)
                    outs.append(model.embed_item_alone(tokens).cpu().numpy())
            self.members.append((model, np.concatenate(outs, axis=0)))
        return time.perf_counter() - t0

    def _query_embeddings_gidx(self, model, base_repr, ctx_gidx_lists):
        """Same math as phase 32 compute_query_embeddings, taking gidx lists."""
        outs = []
        for start in range(0, len(ctx_gidx_lists), OT_QUERY_BATCH):
            chunk = ctx_gidx_lists[start:start + OT_QUERY_BATCH]
            Lmax = max(len(q) for q in chunk)
            B = len(chunk)
            ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
            mask = np.ones((B, Lmax), dtype=bool)  # True = padded
            for i, items in enumerate(chunk):
                for j, g in enumerate(items):
                    ctx_gidx[i, j] = g
                    mask[i, j] = False
            ctx_t = base_repr[torch.tensor(ctx_gidx, device=self.device)]
            mask_t = torch.tensor(mask, device=self.device)
            with torch.no_grad():
                B_, L_, D_ = ctx_t.shape
                tokens = model.encode_item_tokens(ctx_t.reshape(B_ * L_, D_)).reshape(B_, L_, -1)
                q_emb = model.embed_query(tokens, mask_t)
            outs.append(q_emb.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def cir_ranks(self, pools, queries, idx, base_repr):
        """Faithful copy of phase 32 evaluate_recall_ensemble, recording per-query ranks."""
        t0 = time.perf_counter()
        ranks_out = np.full(len(queries), -1, dtype=np.int32)
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
            pool_idx = [idx[i] for i in pool_ids if i in idx]
            kept_qidx, query_item_lists, target_positions = [], [], []
            for qi in qidxs:
                q = queries[qi]
                items = [i for i in q["query_items"] if i in idx]
                if not items or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                kept_qidx.append(qi)
                query_item_lists.append([idx[i] for i in items])
                target_positions.append(pool_pos[q["target_item"]])
            if not query_item_lists:
                continue
            sims_sum = None
            for model, candidate_emb in self.members:
                pool_emb = candidate_emb[pool_idx]
                query_mat = self._query_embeddings_gidx(model, base_repr, query_item_lists)
                sims = query_mat @ pool_emb.T
                sims_sum = sims if sims_sum is None else sims_sum + sims
            sims_avg = sims_sum / len(self.members)
            target_positions_arr = np.array(target_positions)
            target_sims = sims_avg[np.arange(len(query_item_lists)), target_positions_arr]
            ranks = (sims_avg >= target_sims[:, None]).sum(axis=1)
            ranks_out[np.array(kept_qidx)] = ranks
            n_total += len(query_item_lists)
        return ranks_out, n_total, n_skipped, time.perf_counter() - t0

    def score_sets(self, ctx_gidx_lists, cand_gidx_lists, ctx_cat_lists=None, cand_cat_lists=None, base_repr=None, chunk=4096):
        B = len(ctx_gidx_lists)
        A = len(cand_gidx_lists[0])
        out = np.zeros((B, A), dtype=np.float64)
        for start in range(0, B, chunk):
            ctx_chunk = ctx_gidx_lists[start:start + chunk]
            cand_arr = np.asarray(cand_gidx_lists[start:start + chunk], dtype=np.int64)
            acc = None
            for model, candidate_emb in self.members:
                q = self._query_embeddings_gidx(model, base_repr, ctx_chunk)  # (b, 64)
                c = candidate_emb[cand_arr]  # (b, A, 64)
                s = np.einsum("bd,bad->ba", q, c)
                acc = s if acc is None else acc + s
            out[start:start + chunk] = acc / len(self.members)
        return out

    def item_embeddings(self):
        return [cand for _, cand in self.members]


# ================================================================ CSA-NET
class CSASystem:
    """Phase 34's mechanism. Distances (lower = better), length-normalized average
    over context items, category-pair conditioned, averaged across members.
    Candidate embeddings are pool-scoped (P, C, D) per category, as the original;
    the time spent on them is accumulated separately so the cost breakdown is
    comparable with the other two systems."""
    kind = "csa"
    higher_is_better = False

    def __init__(self, name, ckpt_paths, cat_to_idx, item_cat, device=DEVICE):
        self.name, self.ckpt_paths, self.device = name, list(ckpt_paths), device
        self.cat_to_idx, self.item_cat = cat_to_idx, item_cat
        self.C = len(cat_to_idx)
        self.models = None
        self.n_params_per_member = None
        self.precompute_in_loop_s = 0.0

    def precompute(self, base_repr):
        t0 = time.perf_counter()
        self.models = []
        for ckpt in self.ckpt_paths:
            m = load_csa_member(ckpt, self.C, self.device)
            self.n_params_per_member = count_params(m)
            self.models.append(m)
        return time.perf_counter() - t0  # model loading only; candidate embeddings are per pool

    def cir_ranks(self, pools, queries, id_to_gidx, base_repr):
        """Faithful copy of phase 34 train_core.evaluate_recall_ensemble, recording
        per-query ranks (it already tracks kept_qidx)."""
        t0 = time.perf_counter()
        self.precompute_in_loop_s = 0.0
        C = self.C
        cat_to_idx, item_cat_lookup, device, models = self.cat_to_idx, self.item_cat, self.device, self.models
        ranks_out = np.full(len(queries), -1, dtype=np.int32)
        n_total, n_skipped = 0, 0
        by_cat = defaultdict(list)
        for qi, q in enumerate(queries):
            by_cat[q["category"]].append(qi)
        with torch.no_grad():
            for cat, qidxs in by_cat.items():
                if cat not in pools:
                    n_skipped += len(qidxs)
                    continue
                pool_ids = pools[cat]
                pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
                pool_gidx = [id_to_gidx[i] for i in pool_ids if i in id_to_gidx]
                if len(pool_gidx) != len(pool_ids):
                    n_skipped += len(qidxs)
                    continue
                tp = time.perf_counter()
                cand_all_per_member = []
                for m in models:
                    x_pool = m.encode_feature(base_repr[torch.tensor(pool_gidx, device=device)])
                    cat_t_fixed = torch.zeros(len(pool_gidx), C, device=device)
                    cat_t_fixed[:, cat_to_idx[cat]] = 1.0
                    cand_all_per_member.append(m.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed))
                if device == "mps":
                    torch.mps.synchronize()
                self.precompute_in_loop_s += time.perf_counter() - tp

                kept_qidx, ctx_lists, ctx_cat_lists, tpos = [], [], [], []
                for qi in qidxs:
                    q = queries[qi]
                    items = [i for i in q["query_items"] if i in id_to_gidx]
                    if not items or q["target_item"] not in pool_pos:
                        n_skipped += 1
                        continue
                    kept_qidx.append(qi)
                    ctx_lists.append([id_to_gidx[i] for i in items])
                    ctx_cat_lists.append([cat_to_idx[item_cat_lookup[i]] for i in items])
                    tpos.append(pool_pos[q["target_item"]])
                if not kept_qidx:
                    continue
                for start in range(0, len(kept_qidx), CSA_QUERY_CHUNK):
                    sl = slice(start, start + CSA_QUERY_CHUNK)
                    chunk_ctx, chunk_cat, chunk_tpos = ctx_lists[sl], ctx_cat_lists[sl], tpos[sl]
                    B = len(chunk_ctx)
                    Lmax = max(len(c) for c in chunk_ctx)
                    ctx_gidx_np = np.zeros((B, Lmax), dtype=np.int64)
                    ctx_cat_np = np.zeros((B, Lmax), dtype=np.int64)
                    ctx_mask_np = np.zeros((B, Lmax), dtype=bool)
                    for i, (c, cc) in enumerate(zip(chunk_ctx, chunk_cat)):
                        L = len(c)
                        ctx_gidx_np[i, :L] = c
                        ctx_cat_np[i, :L] = cc
                        ctx_mask_np[i, :L] = True
                    ctx_gidx_t = torch.tensor(ctx_gidx_np, device=device)
                    ctx_cat_t = torch.tensor(ctx_cat_np, device=device)
                    ctx_mask_t = torch.tensor(ctx_mask_np, device=device)
                    P = cand_all_per_member[0].shape[0]
                    dist_avg_sum = torch.zeros(B, P, device=device)
                    for m, cand_all in zip(models, cand_all_per_member):
                        x_ctx = m.encode_feature(base_repr[ctx_gidx_t])
                        cat_s_flat = F.one_hot(ctx_cat_t.reshape(-1), C).float()
                        cat_t_flat = torch.zeros(B * Lmax, C, device=device)
                        cat_t_flat[:, cat_to_idx[cat]] = 1.0
                        f_ctx = m.embed_from_feature(x_ctx.reshape(B * Lmax, -1), cat_s_flat, cat_t_flat).reshape(B, Lmax, -1)
                        dist_sum = torch.zeros(B, P, device=device)
                        for c in range(C):
                            sel = (ctx_cat_t == c) & ctx_mask_t
                            if not sel.any():
                                continue
                            sims_c = torch.einsum("bld,pd->blp", f_ctx, cand_all[:, c, :])
                            dist_c = (2.0 - 2.0 * sims_c) * sel.unsqueeze(-1).float()
                            dist_sum += dist_c.sum(dim=1)
                        n_ctx = ctx_mask_t.sum(dim=1, keepdim=True).clamp(min=1).float()
                        dist_avg_sum += dist_sum / n_ctx
                    dist_avg_ensemble = (dist_avg_sum / len(models)).cpu().numpy()
                    tpos_arr = np.array(chunk_tpos)
                    target_dist = dist_avg_ensemble[np.arange(B), tpos_arr]
                    ranks = (dist_avg_ensemble <= target_dist[:, None]).sum(axis=1)
                    ranks_out[np.array(kept_qidx[sl])] = ranks
                    n_total += B
        return ranks_out, n_total, n_skipped, time.perf_counter() - t0

    def score_sets(self, ctx_gidx_lists, cand_gidx_lists, ctx_cat_lists, cand_cat_lists, base_repr=None, chunk=1024):
        """Generic scorer: scores[b, a] = NEGATIVE ensemble-average length-normalized
        squared distance between context b and candidate a, where every (context
        item, candidate) pair is embedded under the SAME (source = context item's
        category, target = candidate's own category) attention weights -- exactly
        the pairing the CIR evaluator uses (there the target category is the pool's
        category, which every pool item shares). Higher = more compatible, so the
        caller can treat all three systems uniformly."""
        C, device, models = self.C, self.device, self.models
        B = len(ctx_gidx_lists)
        A = len(cand_gidx_lists[0])
        out = np.zeros((B, A), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, B, chunk):
                ctx_arr, valid = _pad_gidx(ctx_gidx_lists[start:start + chunk])
                cat_arr, _ = _pad_gidx(ctx_cat_lists[start:start + chunk])
                b, Lmax = ctx_arr.shape
                cand_arr = torch.tensor(np.asarray(cand_gidx_lists[start:start + chunk], dtype=np.int64), device=device)
                cand_cat = torch.tensor(np.asarray(cand_cat_lists[start:start + chunk], dtype=np.int64), device=device)
                ctx_gidx_t = torch.tensor(ctx_arr, device=device)
                ctx_cat_t = torch.tensor(cat_arr, device=device)
                valid_t = torch.tensor(valid, device=device)
                n_ctx = valid_t.sum(dim=1).clamp(min=1).float()
                cat_s_flat = F.one_hot(ctx_cat_t.reshape(-1), C).float()  # (b*Lmax, C)
                dist_acc = torch.zeros(b, A, device=device)
                for m in models:
                    x_ctx = m.encode_feature(base_repr[ctx_gidx_t]).reshape(b * Lmax, -1)
                    x_cand = m.encode_feature(base_repr[cand_arr])  # (b, A, D)
                    for a in range(A):
                        t_onehot = F.one_hot(cand_cat[:, a], C).float()  # (b, C)
                        t_flat = t_onehot.unsqueeze(1).expand(b, Lmax, C).reshape(b * Lmax, C)
                        f_ctx = m.embed_from_feature(x_ctx, cat_s_flat, t_flat).reshape(b, Lmax, -1)
                        x_c = x_cand[:, a, :].unsqueeze(1).expand(b, Lmax, x_cand.shape[-1]).reshape(b * Lmax, -1)
                        f_cand = m.embed_from_feature(x_c, cat_s_flat, t_flat).reshape(b, Lmax, -1)
                        d = (2.0 - 2.0 * (f_ctx * f_cand).sum(dim=-1)) * valid_t.float()  # (b, Lmax)
                        dist_acc[:, a] += d.sum(dim=1) / n_ctx
                out[start:start + chunk] = (-(dist_acc / len(models))).cpu().numpy()
        return out


# ---------------------------------------------------------------- official protocol (verbatim phase 9)
def build_setid_index_resolver(split="test"):
    with open(RAW_DIR / "nondisjoint" / f"{split}.json") as f:
        outfits = json.load(f)
    resolver = {}
    for outfit in outfits:
        set_id = outfit["set_id"]
        for it in outfit["items"]:
            resolver[f"{set_id}_{it['index']}"] = it["item_id"]
    return resolver


def auc_rank_sum(scores, labels):
    """Mann-Whitney U / rank-sum AUC, no sklearn."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    ranks = rankdata(scores)  # average rank for ties
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def load_compatibility_lines(resolver, idx):
    """Parsed once, shared by both protocols. Returns list of (label, [item_id]) and skip count."""
    rows, n_skipped = [], 0
    with open(RAW_DIR / "nondisjoint" / "compatibility_test.txt") as f:
        for line in f:
            parts = line.split()
            label = int(parts[0])
            item_ids = [resolver.get(r) for r in parts[1:]]
            if any(i is None or i not in idx for i in item_ids):
                n_skipped += 1
                continue
            rows.append((label, item_ids))
    return rows, n_skipped


def load_fitb_questions(resolver, idx):
    with open(RAW_DIR / "nondisjoint" / "fill_in_blank_test.json") as f:
        questions = json.load(f)
    rows, n_skipped = [], 0
    for q in questions:
        q_items = [resolver.get(r) for r in q["question"]]
        ans_items = [resolver.get(r) for r in q["answers"]]
        if any(i is None or i not in idx for i in q_items) or any(i is None or i not in idx for i in ans_items):
            n_skipped += 1
            continue
        rows.append((q_items, ans_items))
    return rows, n_skipped


def evaluate_compatibility_pairwise(embeddings_list, idx, compat_rows):
    """Phase 9's protocol generalized to an ensemble by averaging each pair's
    cosine across members (a single-member list is phase 9's code exactly:
    mean pairwise cosine over all C(n,2) pairs)."""
    labels, scores = [], []
    for label, item_ids in compat_rows:
        rows_idx = [idx[a] for a in item_ids]
        pair_sims_sum = None
        for emb in embeddings_list:
            vecs = emb[rows_idx]
            pair_sims = np.array([float(np.dot(vecs[i], vecs[j])) for i, j in combinations(range(len(vecs)), 2)])
            pair_sims_sum = pair_sims if pair_sims_sum is None else pair_sims_sum + pair_sims
        pair_sims_avg = pair_sims_sum / len(embeddings_list)
        score = float(np.mean(pair_sims_avg)) if len(pair_sims_avg) else 0.0
        labels.append(label)
        scores.append(score)
    return auc_rank_sum(scores, labels), len(labels)


def evaluate_fitb_pairwise(embeddings_list, idx, fitb_rows):
    """Phase 9's FITB rule generalized to an ensemble: candidate score = mean
    cosine to the question items, averaged across members; argmax with strict
    `>` (ties resolve to the lower index, as in phase 9). Also counts exact ties
    for transparency."""
    n_correct, n_ties = 0, 0
    for q_items, ans_items in fitb_rows:
        q_idx = [idx[i] for i in q_items]
        scores = np.zeros(len(ans_items))
        for emb in embeddings_list:
            q_vecs = emb[q_idx]
            for ai, ans_item in enumerate(ans_items):
                scores[ai] += float(np.mean(q_vecs @ emb[idx[ans_item]]))
        scores /= len(embeddings_list)
        best_ans, best_score = None, -np.inf
        for ai in range(len(ans_items)):
            if scores[ai] > best_score:
                best_score, best_ans = scores[ai], ai
        if (scores == best_score).sum() > 1:
            n_ties += 1
        if best_ans == 0:
            n_correct += 1
    return n_correct / len(fitb_rows), len(fitb_rows), n_ties


def fmt_pct(x):
    return f"{100 * x:+.1f}%"

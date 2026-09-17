"""
Phase 33, step 0: verify the vectorized compute_batch_loss/evaluate_recall
(train_core.py) reproduce phase 13b's original per-sample/per-negative
Python-loop computation EXACTLY (this is deterministic math, not a sampling
process, so exact numerical equality is the bar -- not a statistical
equivalence test like phase 31's sampler check). Same model instance (same
weights) used for both computations, synthetic data, small enough to run
the original loop-based version directly for comparison.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CSANetSigLIP, pairwise_distance, outfit_ranking_loss, uniformity_loss
import train_core

torch.manual_seed(0)
DEVICE = "cpu"  # exact reproducibility check, not a speed test
NUM_CATEGORIES = 5   # small synthetic category count
N_ITEMS = 40
EMBED_IN = 16
B = 6                # batch of samples
MAX_CTX = 4
MAX_NEG = 3


def onehot_1(idx, C):
    v = torch.zeros(1, C)
    v[0, idx] = 1.0
    return v


def original_compute_batch_loss(model, base_repr, samples, num_categories, margin, uniformity_weight):
    """Faithful re-implementation of phase 13b's train_core.py
    compute_batch_loss (loop-based), operating on the same
    context_gidx/positive_gidx/neg_gidx sample dicts this phase's
    make_sample produces, so it can run directly against the SAME batch
    the vectorized path receives."""
    losses, d_pos_list, d_neg_list, reps = [], [], [], []
    for s in samples:
        n_ctx = len(s["context_gidx"])
        if n_ctx == 0:
            continue
        x_ctx = model.encode_feature(base_repr[s["context_gidx"]])  # (n_ctx, D)
        cat_s_ctx = torch.cat([onehot_1(c, num_categories) for c in s["context_cat_idx"]], dim=0)
        cat_t = onehot_1(s["positive_cat_idx"], num_categories).expand(n_ctx, -1)
        f_ctx = model.embed_from_feature(x_ctx, cat_s_ctx, cat_t)

        x_pos = model.encode_feature(base_repr[s["positive_gidx"]]).unsqueeze(0).expand(n_ctx, -1)
        f_pos = model.embed_from_feature(x_pos, cat_s_ctx, cat_t)
        d_pos_i = pairwise_distance(f_ctx, f_pos)
        D_pos = d_pos_i.mean()
        reps.append(F.normalize(f_pos.mean(dim=0), p=2, dim=-1))

        d_negs = []
        for neg_gidx in s["neg_gidx"]:
            x_neg = model.encode_feature(base_repr[neg_gidx]).unsqueeze(0).expand(n_ctx, -1)
            f_neg = model.embed_from_feature(x_neg, cat_s_ctx, cat_t)
            d_neg_i = pairwise_distance(f_ctx, f_neg)
            d_negs.append(d_neg_i.mean())
        D_negs = torch.stack(d_negs)

        loss = outfit_ranking_loss(D_pos.unsqueeze(0), D_negs.unsqueeze(0), margin=margin, aggregation="min")
        losses.append(loss)
        d_pos_list.append(D_pos.item())
        d_neg_list.append(D_negs.min().item())

    ranking_loss = torch.stack(losses).mean()
    uniformity = uniformity_loss(torch.stack(reps))
    total_loss = ranking_loss + uniformity_weight * uniformity
    return total_loss, float(np.mean(d_pos_list)), float(np.mean(d_neg_list))


def main():
    base_repr = F.normalize(torch.randn(N_ITEMS, EMBED_IN), p=2, dim=-1)
    model = CSANetSigLIP(num_categories=NUM_CATEGORIES, siglip_dim=EMBED_IN).to(DEVICE)
    model.eval()  # no dropout in this model, but eval() for determinism regardless

    rng = np.random.default_rng(1)
    samples = []
    for _ in range(B):
        n_ctx = rng.integers(1, MAX_CTX + 1)
        n_neg = rng.integers(1, MAX_NEG + 1)
        samples.append({
            "context_gidx": rng.integers(0, N_ITEMS, size=n_ctx).tolist(),
            "context_cat_idx": rng.integers(0, NUM_CATEGORIES, size=n_ctx).tolist(),
            "positive_gidx": int(rng.integers(0, N_ITEMS)),
            "positive_cat_idx": int(rng.integers(0, NUM_CATEGORIES)),
            "neg_gidx": rng.integers(0, N_ITEMS, size=n_neg).tolist(),
        })

    orig_loss, orig_dpos, orig_dneg = original_compute_batch_loss(
        model, base_repr, samples, NUM_CATEGORIES, margin=0.3, uniformity_weight=1.0
    )

    batch = train_core.build_batch(samples, DEVICE)
    vec_loss, vec_dpos, vec_dneg = train_core.compute_batch_loss(
        model, base_repr, batch, NUM_CATEGORIES, margin=0.3, uniformity_weight=1.0
    )

    print(f"Original:   loss={orig_loss.item():.8f} D_pos={orig_dpos:.8f} D_neg={orig_dneg:.8f}")
    print(f"Vectorized: loss={vec_loss.item():.8f} D_pos={vec_dpos:.8f} D_neg={vec_dneg:.8f}")

    loss_diff = abs(orig_loss.item() - vec_loss.item())
    dpos_diff = abs(orig_dpos - vec_dpos)
    dneg_diff = abs(orig_dneg - vec_dneg)
    print(f"\nDiffs: loss={loss_diff:.2e} D_pos={dpos_diff:.2e} D_neg={dneg_diff:.2e}")

    TOL = 1e-5
    ok = loss_diff < TOL and dpos_diff < TOL and dneg_diff < TOL
    print("PASS" if ok else "FAIL")
    assert ok, "Vectorized compute_batch_loss does NOT match the original -- do not proceed to real training"

    # --- Gradient check: verify backward() also matches (not just forward values) ---
    model.zero_grad()
    orig_loss.backward()
    orig_grad = model.proj.weight.grad.clone()
    model.zero_grad()
    vec_loss.backward()
    vec_grad = model.proj.weight.grad.clone()
    grad_diff = (orig_grad - vec_grad).abs().max().item()
    print(f"\nGradient check (model.proj.weight): max abs diff = {grad_diff:.2e}")
    assert grad_diff < TOL, "Gradients differ -- vectorized backward pass does not match original"
    print("Gradient check PASS")

    print("\nAll checks passed. Vectorized compute_batch_loss is verified equivalent to the original.")


if __name__ == "__main__":
    main()

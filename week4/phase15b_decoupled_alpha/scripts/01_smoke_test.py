"""
Phase 15b, step 3: confirm phase 15's known safeguards still hold under the
decoupled dual-forward-pass training procedure -- checked directly, not
assumed to carry over just because the architecture is unchanged (only the
training procedure changed, and that's exactly the kind of change that could
plausibly interact with the collapse/dead-gradient failure modes
differently: both loss terms now pull on the shared parameters every single
step instead of alternating/being enveloped, which could in principle change
the collapse dynamics).
"""
from pathlib import Path

import numpy as np
import torch

from model import CSANetSigLIPControllable
from train_core import run_training_decoupled, WEIGHT_SUB

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"
PHASE15_DIR = BASE_DIR.parent / "phase15_controllable_subspace_attention"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"
NEGATIVE_CANDIDATES_JSON = PHASE13_DIR / "data" / "negative_candidates.json"
SAME_CAT_NEIGHBORS_NPZ = PHASE15_DIR / "data" / "same_category_neighbors.npz"

DEVICE = "cpu"  # matches phase 15's measured finding: CPU faster than MPS for this workload
N_TRAIN_OUTFITS = 2000
N_VAL_OUTFITS = 400
MAX_EPOCHS = 5
BATCH_SIZE = 96


def mean_pairwise_cosine(model, embeddings, n_samples=2000):
    rng = np.random.default_rng(0)
    sample = rng.choice(len(embeddings), size=min(n_samples, len(embeddings)), replace=False)
    vecs = torch.tensor(embeddings[sample], device=DEVICE)
    x = model.encode_feature(vecs)
    C = model.num_categories
    cat_fixed = torch.zeros(len(sample), C, device=DEVICE)
    cat_fixed[:, 0] = 1.0
    with torch.no_grad():
        f = model.embed_from_feature(x, cat_fixed, cat_fixed, alpha=0.5)
        sims = f @ f.T
        n = sims.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
        return sims[mask].mean().item()


def gradient_sanity_check():
    """Dead-gradient check: one backward pass on a real batch, untrained
    model, confirm every shared param gets a nonzero gradient."""
    from train_core import TrainState, make_batch, compute_batch_loss_decoupled

    state = TrainState(EMBEDDINGS_NPZ, TRAINING_DATA_JSON, NEGATIVE_CANDIDATES_JSON,
                        SAME_CAT_NEIGHBORS_NPZ, seed=0)
    model = CSANetSigLIPControllable().to(DEVICE)
    samples, substitute_pairs, vecs, gidx_to_pos = make_batch(
        state, state.train_outfits, list(range(96)), "train", DEVICE)
    x_all = model.encode_feature(vecs)
    loss, _, _, _, _ = compute_batch_loss_decoupled(
        model, state, samples, substitute_pairs, x_all, gidx_to_pos, DEVICE, weight_sub=WEIGHT_SUB)
    loss.backward()

    results = []
    for name, p in model.named_parameters():
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        results.append((name, gn))
        print(f"  {name}: grad_norm={gn:.6f}")
    n_dead = sum(1 for _, gn in results if gn == 0.0)
    return results, n_dead


def main():
    lines = ["# Phase 15b: Smoke Test Report", ""]

    lines.append("## Dead-gradient check (untrained model, one real 96-outfit batch)")
    lines.append("")
    results, n_dead = gradient_sanity_check()
    for name, gn in results:
        lines.append(f"- `{name}`: grad_norm={gn:.6f}")
    lines.append("")
    if n_dead == 0:
        lines.append("**No dead gradients** -- every shared parameter (including `attn_net`'s "
                      "new alpha-input weights, unchanged from phase 15) receives a nonzero "
                      "gradient at init under the decoupled dual-forward-pass loss.")
    else:
        lines.append(f"**{n_dead} parameter(s) have zero gradient** -- investigate before trusting "
                      "the real run.")
    lines.append("")

    lines.append("## Collapse check (with vs. without uniformity regularizer)")
    lines.append("")
    lines.append(f"5 epochs, {N_TRAIN_OUTFITS} train outfits, {N_VAL_OUTFITS} val outfits, "
                  f"weight_sub={WEIGHT_SUB} (reused from phase 15's calibration), decoupled "
                  "dual-forward-pass loss (complement fixed at alpha=0, substitute fixed at alpha=1, every step).")
    lines.append("")

    for uniformity_weight, label in [(0.0, "no_uniformity"), (1.0, "with_uniformity")]:
        print(f"\n=== Collapse check: uniformity_weight={uniformity_weight} ({label}) ===")
        hist = run_training_decoupled(
            EMBEDDINGS_NPZ, TRAINING_DATA_JSON, NEGATIVE_CANDIDATES_JSON, SAME_CAT_NEIGHBORS_NPZ,
            out_dir=BASE_DIR / "data" / f"smoke_{label}", device=DEVICE,
            max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=5e-5, patience=10,
            weight_sub=WEIGHT_SUB, n_train_outfits=N_TRAIN_OUTFITS, n_val_outfits=N_VAL_OUTFITS,
            log_every=999999, uniformity_weight=uniformity_weight,
        )
        data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        model = CSANetSigLIPControllable().to(DEVICE)
        model.load_state_dict(torch.load(BASE_DIR / "data" / f"smoke_{label}" / "csa_net_decoupled_best.pt",
                                          map_location=DEVICE))
        model.eval()
        cos = mean_pairwise_cosine(model, embeddings)
        print(f"mean pairwise candidate-embedding cosine similarity ({label}): {cos:.4f}")
        lines.append(f"- {label}: mean pairwise cosine similarity = {cos:.4f}")

    lines.append("")
    lines.append("Uniformity regularizer enabled from the start of the real run per phases "
                  "13b/14/15's proactive convention regardless of this check's outcome, unless "
                  "it shows the regularizer is actively harmful (not expected).")

    (BASE_DIR / "training_log_smoke_test.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {BASE_DIR / 'training_log_smoke_test.md'}")


if __name__ == "__main__":
    main()

"""
Phase 13b, step 4 prep: compute the trained model's base feature `x` (the
projected 64-d feature BEFORE category-conditioned masking) for all 251,008
items. Unlike phase 13, this needs no GPU/Modal round trip -- `encode_feature`
is just a normalize + Linear(768, 64), applied directly to phase 9's already-
loaded SigLIP matrix in one shot, seconds on CPU.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CSANetSigLIP

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT_PT = BASE_DIR / "models" / "csa_net_siglip_best.pt"
OUT_NPZ = BASE_DIR / "embeddings" / "csa_siglip_base_features.npz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = torch.tensor(data["embeddings"].astype(np.float32), device=DEVICE)
    print(f"Loaded {len(item_ids)} SigLIP embeddings.")

    model = CSANetSigLIP().to(DEVICE).eval()
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)

    with torch.no_grad():
        x = model.encode_feature(embeddings).cpu().numpy()

    np.savez(OUT_NPZ, item_ids=np.array(item_ids), embeddings=x.astype(np.float32))
    print(f"Saved {x.shape} base features to {OUT_NPZ}")


if __name__ == "__main__":
    main()

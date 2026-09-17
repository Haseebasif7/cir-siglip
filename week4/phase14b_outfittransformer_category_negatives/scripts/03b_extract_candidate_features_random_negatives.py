"""
Phase 14b, ablation run: same as 03_extract_candidate_features.py, pointed
at the random-same-category-negatives checkpoint (models_random_negatives/)
instead of the mined-negatives one.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT_PT = BASE_DIR / "models_random_negatives" / "outfit_transformer_siglip_random_neg_best.pt"
OUT_NPZ = BASE_DIR / "embeddings" / "outfit_transformer_candidate_features_random_negatives.npz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 4096


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = torch.tensor(data["embeddings"].astype(np.float32), device=DEVICE)
    print(f"Loaded {len(item_ids)} SigLIP embeddings.")

    model = OutfitTransformerSigLIP().to(DEVICE).eval()
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)

    outs = []
    with torch.no_grad():
        for start in range(0, len(item_ids), BATCH):
            chunk = embeddings[start:start + BATCH]
            tokens = model.encode_item_tokens(chunk)
            emb = model.embed_item_alone(tokens)
            outs.append(emb.cpu().numpy())
    candidate_features = np.concatenate(outs, axis=0)

    np.savez(OUT_NPZ, item_ids=np.array(item_ids), embeddings=candidate_features.astype(np.float32))
    print(f"Saved {candidate_features.shape} candidate features to {OUT_NPZ}")


if __name__ == "__main__":
    main()

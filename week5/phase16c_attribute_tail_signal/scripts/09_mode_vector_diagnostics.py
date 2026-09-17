"""Same mechanistic verification phase 16 used: learned mode-vector norms
and their cosine similarity to each other, relative to the shared base
projection's typical norm. Phase 16's diagnosis was that its mode vectors
stayed small (0.191/0.166 vs base ~1.233) and similar to each other
(cosine 0.55) -- this check is the direct test of whether phase 16c's
genuinely different training populations pushed the two modes further
apart than reweighting alone did.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
CHECKPOINT = BASE_DIR / "models" / "attribute_tail_dial.pt"
GALLERY_NPZ = BASE_DIR.parent / "phase16_relevance_tail_dial" / "data" / "candidate_gallery.npz"
OUT_MD = BASE_DIR / "logs" / "mode_vector_diagnostics.md"

DEVICE = "cpu"

# phase 16's own numbers, for direct comparison
PHASE16_REL_NORM = 0.1914
PHASE16_TAIL_NORM = 0.1660
PHASE16_COSINE = 0.5525
PHASE16_BASE_NORM = 1.2334


def main():
    m = ControllableProjectionHead().to(DEVICE)
    m.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    m.eval()

    rel_norm = m.mode_relevance.norm().item()
    tail_norm = m.mode_tail.norm().item()
    diff_norm = (m.mode_relevance - m.mode_tail).norm().item()
    cos = torch.nn.functional.cosine_similarity(
        m.mode_relevance.unsqueeze(0), m.mode_tail.unsqueeze(0)).item()

    d = np.load(GALLERY_NPZ, allow_pickle=True)
    emb = d["embeddings"][:2000]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = (emb / norms).astype("float32")
    x = torch.tensor(emb)
    with torch.no_grad():
        base = m.net(x)
    base_mean_norm = base.norm(dim=-1).mean().item()
    base_std_norm = base.norm(dim=-1).std().item()

    lines = ["# Phase 16c: Mode Vector Mechanistic Diagnostics\n"]
    lines.append("Same check phase 16 used, direct comparison:\n")
    lines.append("| | Phase 16 | Phase 16c |")
    lines.append("|---|---|---|")
    lines.append(f"| mode_relevance norm | {PHASE16_REL_NORM:.4f} | {rel_norm:.4f} |")
    lines.append(f"| mode_tail norm | {PHASE16_TAIL_NORM:.4f} | {tail_norm:.4f} |")
    lines.append(f"| cosine(mode_relevance, mode_tail) | {PHASE16_COSINE:.4f} | {cos:.4f} |")
    lines.append(f"| base projection mean norm | {PHASE16_BASE_NORM:.4f} | {base_mean_norm:.4f} (std {base_std_norm:.4f}) |")
    lines.append("")
    rel_pct = rel_norm / base_mean_norm * 100
    tail_pct = tail_norm / base_mean_norm * 100
    lines.append(f"Mode vector magnitude relative to base projection: mode_relevance is {rel_pct:.1f}% of the "
                 f"base norm, mode_tail is {tail_pct:.1f}% (phase 16: {PHASE16_REL_NORM/PHASE16_BASE_NORM*100:.1f}% "
                 f"/ {PHASE16_TAIL_NORM/PHASE16_BASE_NORM*100:.1f}%).\n")
    if cos < PHASE16_COSINE - 0.1:
        lines.append(f"**The two mode vectors point in more DIVERGENT directions than phase 16's did** "
                     f"(cosine {cos:.4f} vs phase 16's {PHASE16_COSINE:.4f}) -- direct mechanistic evidence "
                     "that training on genuinely different edge populations pushed the modes further apart "
                     "than reweighting the same population did.")
    elif cos > PHASE16_COSINE + 0.1:
        lines.append(f"**The two mode vectors point in a MORE similar direction than phase 16's did** "
                     f"(cosine {cos:.4f} vs phase 16's {PHASE16_COSINE:.4f}) -- despite training on genuinely "
                     "different populations, this did not translate into more divergent mode vectors, worth "
                     "flagging as a real, unexpected finding rather than assuming the fix worked mechanistically.")
    else:
        lines.append(f"**Cosine similarity ({cos:.4f}) is comparable to phase 16's ({PHASE16_COSINE:.4f})** -- "
                     "no strong mechanistic shift in how divergent the two mode vectors are, despite training "
                     "on genuinely different populations this time.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"rel_norm={rel_norm:.4f} tail_norm={tail_norm:.4f} cosine={cos:.4f} base_mean_norm={base_mean_norm:.4f}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()

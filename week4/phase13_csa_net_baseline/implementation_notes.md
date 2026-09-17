# Phase 13, Step 1-2: What's Available, and How CSA-Net Was Implemented

## Step 1: what's actually available

**Paper.** Yen-Liang Lin, Son Tran, Larry S. Davis (Amazon), "Fashion Outfit
Complementary Item Retrieval," CVPR 2020. (The brief's placeholder author
list "Lin, Kovacs, Bell, Davis" was not correct -- the real byline is Lin,
Tran, Davis, confirmed directly from the paper PDF, arXiv:1912.08967.)
openaccess.thecvf.com blocked automated fetches (403); the arXiv mirror
(https://arxiv.org/abs/1912.08967, PDF at /pdf/1912.08967) is the identical
CVPR camera-ready and was used instead, read in full (all 8 pages).

**Unofficial implementations, checked directly rather than trusting
descriptions (same scrutiny phase 12 applied to the OutfitTransformer repo):**

- `github.com/owj0421/csa-net` -- the repo the brief specifically named. It
  exists and is explicitly titled as a CSA-Net implementation, but its own
  README states **"This project is under construction"** and the "Train" /
  "Test" sections are empty placeholders. It has `src/`, `environment.yml`,
  dataset-download instructions, but no working model code, no loss
  function, no evaluation script, and no checkpoint. **Not usable as a
  reference implementation** -- confirmed by reading the actual repo
  content, not just the README's framing.
- `github.com/CrossmodalGroup/CSA-Net` -- name coincidence only. This is
  "Cascade Semantic Prompt Alignment for Image Captioning" (COCO/NoCaps),
  completely unrelated to fashion.
- `github.com/mirthAI/CSA-Net` -- name coincidence only. This is a medical
  image segmentation model for prostate MRI, completely unrelated.

**Conclusion: no usable unofficial implementation exists.** The
architecture, loss, and training procedure below were implemented directly
from the paper's own equations and stated hyperparameters (Sections 3.1,
3.2, 4.2), not adapted from any existing code.

## Step 2: architecture, as implemented (`scripts/model.py`)

Everything below traces to a specific paper statement; assumptions the
paper left unstated are flagged explicitly.

| Component | Paper source | Implementation |
|---|---|---|
| Backbone | "We use ResNet18 [6] as our backbone CNN model... pre-trained on Imagenet" (4.2) | `torchvision.models.resnet18(weights=IMAGENET1K_V1)`, `fc` replaced with `Identity()` (keep 512-d pooled feature) |
| Embedding size | "embedding size 64 similar to state-of-the-art methods... for fair comparison" (4.2) | `Linear(512, 64)` projection on top of the ResNet18 feature |
| Subspaces (k) | "We set the number of subspaces to 5 as in [15]" (4.2) | `NUM_SUBSPACES = 5`, `k` learnable 64-d masks `m_1..m_5`, `nn.Parameter` |
| Category vector | "e.g., 11 semantic categories in Polyvore-Outfit dataset" (3.3) | one-hot over the 11 `semantic_category` labels already established as this project's vocabulary in phases 9/12 (shoes, jewellery, bags, tops, bottoms, all-body, outerwear, sunglasses, accessories, hats, scarves) |
| Attention sub-network | "a sub-network, which contains two fully connected layers and a soft-max layer at the end" (3.1), input = concat of two one-hot category vectors | `Linear(22, H) -> ReLU -> Linear(H, 5) -> softmax`. **Assumption**: hidden width `H` is not stated in the paper -- set to 64 (= embedding size) as a reasonable default, not tuned. |
| Final embedding | `f = sum_i (x ⊙ m_i) * w_i` (eq. 1) | implemented exactly: elementwise mask product, weighted sum over subspaces |
| Distance metric | "pairwise distance" `d(.,.)`, not specified beyond that (3.2, eq. 5) | squared Euclidean distance -- the standard choice in this line of work (Type-aware, SCE-Net both use it); **assumption**, paper doesn't name it explicitly |
| Outfit ranking loss | `D_outfit(O,s) = (1/n) sum_i d(f_i^o, f_i^s)`, `l(O,p,N) = max(0, D_p - D_N + m)` (eq. 5-7) | implemented exactly, including the `varphi` negative-aggregation function |
| Negative aggregation | Table 3: "min" aggregation beats "average" | `aggregation="min"` used throughout (paper's own best setting) |
| Margin | "We set the margin to 0.3" (4.2) | `MARGIN = 0.3` |
| Optimizer / LR | "optimize using ADAM... initial learning rate to 5e-5... linearly decreases the learning rate to zero... warmup ratio to zero" (4.2) | `torch.optim.Adam(lr=5e-5)` + `LinearLR(start_factor=1.0, end_factor=0.0)`, no warmup |
| Batch size | "mini-batch size of 96" (4.2) | `BATCH_SIZE = 96`, one training sample = one outfit (leave-one-out positive + context) |
| Negative sampling | "randomly sample a set of negative images that have the same category as the positive image... select semi-hard negative images" (4.2) | see below -- the one place a real, documented compute-budget deviation was needed |

### The one deliberate deviation: negative sampling

The paper says negatives are same-category and "semi-hard" but does not
state (a) how many negatives per training sample, or (b) how the semi-hard
subset is selected before the margin comparison. True online semi-hard
mining as commonly implemented would require re-embedding the ENTIRE
same-category pool (up to ~51,000 items for shoes) through the
still-training CNN at every single training step to find margin-satisfying
candidates -- not tractable within this project's compute budget.

**What was implemented instead** (`scripts/02_mine_negative_candidates.py`):
a one-time, offline precomputation of each item's top-20 same-category
nearest neighbors by **raw SigLIP cosine similarity** (phase 9's own
already-computed `embeddings/siglip_base.npz`, a fixed general-purpose
visual space, unrelated to CSA-Net's own embeddings), capped at 0.97
similarity to exclude near-duplicate images. This is the *exact same
convention* phase 9 already used for its own hard-negative mining
(`04_mine_hard_negatives.py`), extended here with a same-category
restriction CSA-Net's loss requires but phase 9's MNRL setup didn't.

At training time, each sample's negatives (`NUM_NEGATIVES = 10` -- also
unspecified in the paper, chosen as a reasonable default) are a random
subset of the positive's mined top-20 list (falling back to random
same-category items if fewer than 10 mined candidates exist). This is
"semi-hard" only in the loose sense of "visually plausible, not random" --
not the paper's literal within-margin online definition. **This is the
single biggest fidelity gap in this reproduction and is flagged here
explicitly rather than silently**; if the reproduction's numbers land far
from the published ones, this is the first place to suspect.

### Evaluation protocol adaptation (needed to run under this project's own harness)

Phase 12's `evaluate_recall()` assumes one fixed embedding per item, scored
by a single mean-pooled query vector's dot product -- this doesn't fit
CSA-Net, whose entire contribution is that an item's embedding depends on
which category it's being compared against. Forcing CSA-Net into a
single-vector scheme would defeat the mechanism being reproduced, so
`scripts/04_csa_cir_eval.py` implements CSA-Net's own scoring rule (paper
eq. 5: average per-context-item pairwise distance) directly, while reusing
phase 12's exact same benchmark file (`cir_benchmark.json` -- same pools,
same leave-one-out queries) and the same recall@k bookkeeping (rank of the
true target among all pool candidates, `rank <= k`). Only the
similarity/distance computation differs, and it differs necessarily, not as
a loosening of the comparison. See `model.py`'s
`all_as_candidate_embeddings_from_feature` / `all_category_embeddings_from_feature`
docstrings for the exact indexing scheme this required (a direct
implementation of the paper's own Section 3.3 "enumerate target categories"
indexing idea, adapted to the two different roles context items vs.
candidates play in eq. 2-4).

### Compute: why Modal GPU was used

Full end-to-end ResNet18 fine-tuning across 53,306 training outfits (each
requiring ~15 unique image forward+backward passes per sample at batch 96)
is far beyond what local MPS compute can do in reasonable time -- this
project's own standing convention (phase 9 used Modal for SigLIP extraction
alone, which is a much cheaper frozen-inference-only workload). Training run
via `scripts/modal_train_csa_net.py` on Modal `A10G`, reusing the
already-populated `phase9-polyvore-images` volume (same 251,008-image tar
used by phase 9/10/13's own negative mining) -- no re-upload needed.

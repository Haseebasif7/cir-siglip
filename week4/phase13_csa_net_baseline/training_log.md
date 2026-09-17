# Phase 13, Step 3: Training Log

## Procedure actually used

Full details and paper-vs-implementation trace in `implementation_notes.md`;
this file logs what actually happened when it ran.

| Setting | Value | Source |
|---|---|---|
| Backbone | ResNet18, ImageNet-pretrained, fine-tuned end-to-end | paper 4.2 |
| Embedding size | 64 | paper 4.2 |
| Subspaces (k) | 5 | paper 4.2 |
| Margin | 0.3 | paper 4.2 |
| Negative aggregation | min | paper Table 3 (best setting) |
| Optimizer | Adam, initial LR 5e-5, linear decay to 0, no warmup | paper 4.2 |
| Batch size | 96 outfits | paper 4.2 |
| Negatives per sample | 10, drawn from mined top-20 same-category SigLIP neighbors | **deviation, see implementation_notes.md** -- paper doesn't state a count or exact mining mechanism |
| Max epochs / patience | 20 / 3 (early stopping on validation outfit-ranking loss) | matches this project's own convention (phases 7-12) |
| Training data | Official Polyvore Outfits nondisjoint split (53,306 train outfits, 5,000 val outfits), leave-one-out per outfit, re-randomized each epoch | phase 9's already-downloaded official splits |
| Compute | Modal `A10G` GPU, `cpu=8` (image loading is the non-GPU bottleneck) | standing project convention for CNN-scale workloads |

## What actually happened

Six real Modal GPU training attempts were needed before training was numerically
stable. Five real, reproducible optimization failures were found and fixed in
sequence -- each confirmed on the ACTUAL 251,008-item dataset (not just guessed
from theory), not treated as fixed until the fix was verified to hold at full
scale:

1. **CUDA OOM** (attempt 1): batch=96 outfits, each requiring ~15 unique
   context/positive/negative images, meant ~1,500 images went through ResNet18
   WITH gradients in a single forward+backward -- tried to allocate >20GB on
   the A10G's 24GB. **Fix**: gradient accumulation -- each 96-outfit logical
   batch is split into 12-outfit micro-batches, each backward()-accumulated
   with a loss scaled to its share of the batch, one `optimizer.step()` after
   the full batch. Mathematically identical averaged gradient to a true
   batch-96 step, bounded peak memory.
2. **Dead gradients** (attempt 2): masks initialized `randn * 0.01` (a
   reasonable-looking small-noise default) made every embedding `f = sum_i
   (x*m_i)*w_i` near-zero at init regardless of the input image, since the
   softmax attention weights sum to 1 but the masks themselves killed the
   signal. Confirmed directly: masks grad norm ~0.007 at init, loss stuck at
   EXACTLY the margin value (0.3000) for a full epoch (555 steps), D_pos/D_neg
   both ~0.0000. **Fix**: near-identity mask init (`ones + randn*0.01`) --
   makes `f ~= x` at init, giving the pretrained backbone's own
   already-somewhat-discriminative features as the starting point. Verified:
   masks grad norm jumped to ~2.17 (~300x), loss started moving substantially.
3. **Magnitude collapse** (attempt 3, same run continued after the mask fix):
   with real gradient signal now present, distances (D_pos, D_neg) shrank
   together from ~13-27 at step 50 toward ~0.02-0.04 by the end of epoch 1 --
   the un-normalized hinge loss has a trivial degenerate solution (shrink
   every embedding toward the origin, driving both distances to 0 together,
   satisfying `D_pos - D_neg + margin <= 0` without learning anything).
   **Fix**: L2-normalize the final embedding `f` before computing distance --
   not in the paper's own equation 1 explicitly, but standard practice in the
   closely related papers CSA-Net cites (Type-aware, SCE-Net) and in every
   other phase of this project (7-12). Bounds squared-L2 distance to [0,4],
   removing the magnitude-collapse escape route.
4. **Direction collapse** (attempt 4): even normalized, D_pos and D_neg still
   shrank together on the FULL dataset (0.31 -> 0.04 over one epoch),
   converging toward margin=0.3 again -- confirmed as genuine collapse, not
   just tight clustering, by converting distance to cosine similarity:
   D=0.043 for unit vectors implies cosine similarity ~0.98 between
   essentially unrelated items. Freezing the pretrained backbone for the
   first 3 epochs (attempt 5, a standard transfer-learning safeguard) did
   NOT fix this -- the still-randomly-initialized proj/mask/attention layers
   collapsed the embedding space on their own, independent of backbone
   drift. **Fix**: added Wang & Isola's (2020) alignment/uniformity
   regularizer on top of (not instead of) the paper's own hinge loss --
   `log(mean(exp(-t*||f_i-f_j||^2)))` over a batch's representative
   embeddings, weight 1.0. This term is MINIMIZED when embeddings spread
   across the hypersphere and MAXIMIZED (worst, =0) exactly at collapse,
   removing the free lunch without changing what a genuinely
   compatibility-discriminative embedding looks like. Verified directly
   (collapsed synthetic embeddings score 0.0, random-direction synthetic
   embeddings score -3.06) and then on real data at full scale: D_pos/D_neg
   held stable around 1.95-2.0/1.8 across 250+ steps with no drift toward
   collapse. **This is the one clearly non-paper-specified addition to the
   loss function itself; every other fix changed only initialization,
   normalization, or training schedule, not the objective.**

## Actual training run (attempt 6, `csa_net_best.pt`)

Trained on Modal `A10G` + `cpu=8`. Ran stably for **5 completed epochs**
before being stopped early (see "compute budget" below):

| Epoch | train_loss | val_loss | val_D_pos | val_D_neg | D_pos - D_neg gap |
|---|---|---|---|---|---|
| 0 | -3.212 | -3.328 | 1.974 | 1.812 | 0.162 |
| 1 | -3.356 | -3.365 | 1.971 | 1.814 | 0.157 |
| 2 | -3.379 | -3.374 | 1.967 | 1.813 | 0.154 |
| 3 (backbone unfrozen) | -3.544 | -3.386 | 1.941 | 1.822 | 0.119 |
| 4 | -3.610 | -3.395 | 1.936 | 1.831 | 0.106 |

val_loss improved every single epoch (new best checkpoint saved each time --
no early-stopping plateau was ever reached). The gap between D_pos and
D_neg (positive item's average distance to context vs. the hardest of 10
mined negatives) narrowed steadily and specifically accelerated right after
the backbone unfroze at epoch 3 (0.154 -> 0.119, the largest single-epoch
drop) -- a real signal that genuine compatibility structure was starting to
develop, not just the uniformity term coasting. **Note the sign: D_pos >
D_neg throughout all 5 epochs** -- the model has NOT yet learned to rank the
true compatible item closer than its hardest mined negative on average; it
was still converging toward that when training was stopped.

## Compute budget (real constraint, not a paper deviation)

At this scale (~40-45 min/epoch on A10G, ~1,500 images/step with gradient
accumulation), the configured 20-epoch schedule would need roughly 13-15
hours of GPU time. The user's available Modal budget ($9 remaining,
partially depleted by the five failed attempts above plus other phases'
earlier Modal usage) could not cover that. **By explicit user decision**,
training was stopped after 5 epochs (~4 hours total GPU time for this run)
rather than risk running out of budget mid-run or overdrafting. The
`csa_net_best.pt` checkpoint (epoch 4, best validation loss) is therefore
from an UNDERTRAINED model relative to what the paper's own setup likely
used -- this is the central caveat for interpreting `results_table.md` and
is addressed directly in `phase13_notes.md`.

# Parameter-Ratio Verification: What the Published Pipelines Actually Fine-Tune

Why: `paper/paper_framing_direction.md` asserted "roughly two orders of magnitude fewer trainable parameters than
the published fully fine-tuned pipeline" and flagged it for verification before print. Verified here from the
papers directly (ar5iv renderings of arXiv 2204.04812 and arXiv 1912.08967, fetched 2026-09-15). **The claim does
not survive: the ratio is roughly one order of magnitude (about 7-9x), not two.** Neither paper states a parameter
count, so the figures below are estimates from the stated architectures, with every assumption listed.

## What each paper says it trains

**OutfitTransformer (Sarkar et al., WACV 2023, arXiv 2204.04812), Section 3.3:**
> "We finetune the weights of the image encoder and the fc layer of the text encoder."

- Image encoder: **ResNet-18**, ImageNet-pretrained, fine-tuned end-to-end. Table 3 of the paper reports AUC 0.82
  with the ResNet-18 frozen vs. 0.91 fine-tuned end-to-end -- the authors themselves measure fine-tuning as worth
  +0.09 AUC on their backbone.
- Text encoder: pre-trained **SentenceBERT**, of which **only the added fc layer** is trained; the SentenceBERT body
  is frozen.
- Set encoder: "a six-layer transformer encoder with 16 heads", 128-d item tokens (64 image + 64 text). The
  feed-forward width is not stated.
- Training: batch 50, lr 1e-5 halved every 10 epochs, margin 2, 10 negatives per outfit.

**CSA-Net (Lin et al., CVPR 2020, arXiv 1912.08967):**
- Image encoder: **ResNet-18**, ImageNet-pretrained, "trained in an end-to-end manner".
- No text: "Note that our method does not use text feature."
- Head: 5 subspaces, 64-d embedding, attention = "two fully connected layers and a soft-max layer" (widths unstated).
- Training: batch 96, lr 5e-5 linearly decayed, margin 0.3, random same-category negatives **plus semi-hard
  negative selection** for the outfit ranking loss.

## Trainable-parameter estimates

| Component | Count | Basis |
|---|---|---|
| ResNet-18 convolutional trunk (excl. ImageNet 1000-way fc) | ~11.18 M | torchvision ResNet-18 = 11,689,512 total, of which the 512x1000 fc is 513,000 |
| ResNet-18 -> 64-d image fc | 32,832 | 512x64 + 64 |
| SentenceBERT -> 64-d text fc (OT only) | 49,216 | 768x64 + 64 (SentenceBERT body frozen per the paper) |
| OT transformer encoder, 6 layers, d=128, 16 heads | 1.19 M -- 3.56 M | per layer: attention 66,048 + LayerNorms 512 + FFN; FFN = 131,584 if dim_ff=512 (4x d) or 526,464 if dim_ff=2048 (PyTorch default); width unstated, so a range |
| OT outfit token + heads | < 0.01 M | negligible |
| CSA-Net masks (5 x 64) + attention MLP | ~2 k | negligible |

| Pipeline | Estimated trainable parameters | Ratio to ours (1,705,088) | log10 |
|---|---|---|---|
| **OutfitTransformer, published** | **~12.5 M -- 14.8 M** | **7.3x -- 8.7x** | 0.86 -- 0.94 |
| **CSA-Net, published** | **~11.2 M** | **6.6x** | 0.82 |
| Ours (phase 28 head; frozen SigLIP) | 1.71 M | 1x | -- |
| *For reference: ours vs. our own frozen-backbone CSA-Net reproduction (100,485)* | -- | *ours is 17x larger* | -- |

Assumptions that move the estimate: whether the ImageNet fc is counted (it would add 0.5 M to both published
pipelines -- immaterial to the order of magnitude); the OT transformer's feed-forward width (the single largest
uncertainty, bounded above); whether any part of SentenceBERT beyond the fc was tuned (the paper says fc only). No
plausible resolution of these reaches 100x.

## The sentence that should replace the framing direction's claim

> Our frozen-backbone system trains roughly **one order of magnitude fewer parameters** (about 7-9x) than the
> published pipelines, which fine-tune a ResNet-18 end-to-end. The reduction comes from freezing the backbone,
> not from a smaller head: our head is the largest of the three matched-condition heads (1.7 M vs. 1.0 M and
> 0.1 M).

The argument the paper actually needs -- that a strong frozen general-purpose representation with a light head
matches or beats pipelines that fine-tuned a task-specific backbone -- does not depend on the larger number, and
now has like-for-like support on the released split (see `official_benchmark.md`). Two bounds ride along, per the
framing direction: SigLIP is a much stronger pretrained model than ResNet-18, and SigLIP fine-tuning was never
tested in this project.

## Action taken

`paper/paper_framing_direction.md`'s "roughly two orders of magnitude" sentence is corrected in this phase to
"roughly one order of magnitude (about seven- to nine-fold; see phase 36 `parameter_ratio_verification.md`)".

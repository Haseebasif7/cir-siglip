# Phase 9: Compatibility Projection Trained on Polyvore Outfits — Notes

## Context

Phase 8 showed a real but capped result on Amazon `also_buy` data: only 21.9% of edges are even cross-category, and that's before asking whether those pairs are genuine style complements or incidental co-purchases. This phase tests the same architecture (single-variable test, per the brief) on Polyvore Outfits — 68,306 real human-assembled outfits, purpose-built for compatibility prediction — to isolate whether phase 7/8's ceiling was about data quality specifically.

## Step 0: the dataset was gated, resolved by the user

`mvasil/polyvore-outfits` on Hugging Face is gated (auto-approved, but requires a logged-in HF account + valid token). Per the brief's own instruction, this was reported and the user resolved it directly (requested access, provided a fresh token). No workaround was attempted.

## Step 1: structure — confirmed, not assumed

Full detail in `dataset_structure_check.md`. Key findings that shaped everything downstream: 68,306 outfits (53,306/5,000/10,000 train/valid/test, exactly matching the paper), 251,008 items each with a directly-usable `semantic_category` field (11 values), and — the one real surprise — **no flat `images/` folder in the HF repo**; images are packed inside `data/nondisjoint/{train,valid,test}.parquet` as `{item_id, image:{bytes,path}}` rows, unlike the original GitHub repo's expected layout. Decoded all 251,008 unique images (deduplicated across splits) to local jpgs before anything else could proceed.

## Step 2: positive pairs — the brief's assumption confirmed strongly

76,293 → this phase's 1,373,702 directed positive edges (every pairwise combination within each official train outfit, both directions, official train/valid split used directly rather than a re-derived random split). **95.6% of these pairs are cross-category**, vs. Amazon's 74.1% *same*-category (phase 8). This single number is the clearest confirmation of the phase's premise: Polyvore's co-outfit data is overwhelmingly the kind of signal phase 7/8 was trying and failing to extract from Amazon's `also_buy` graph.

## Step 3-4: extraction and training — infrastructure notes worth keeping

SigLIP extraction for 251,008 images was moved to a Modal L4 GPU mid-phase (local MPS was projected at 3-4 hours; a first Modal attempt reading images from a Modal Volume network filesystem was even slower, ~27 hours projected, because per-file network latency dominates at this many small files — fixed by extracting the uploaded tar to the container's **local** disk instead, dropping the actual run to ~28 minutes). Two smaller bugs hit and fixed along the way: a missing `sentencepiece` dependency in the Modal image, and macOS `tar` silently bundling `._*` AppleDouble sidecar files that aren't real images (filtered by glob everywhere images are listed). Training (1.37M edges, ~6-7 min/epoch) converged for both models, but **the best checkpoint for both was epoch 0** — val loss rose every epoch after that. With ~10,732 batches/epoch here (vs. phase 8's ~118), the true optimum plausibly sits within epoch 0 itself; per-epoch checkpointing is coarse at this scale. Not a bug (early stopping worked exactly as designed and saved the right checkpoint), just a methodological note for any future phase training at this scale.

## Step 5.1: official Polyvore benchmark — a dramatic, unambiguous win

| Configuration | Compatibility AUC | FITB Accuracy |
|---|---|---|
| Raw SigLIP | 0.717 | 0.484 |
| Model A (random negs) | **0.947** | **0.703** |
| Model B (hard negs) | 0.937 | 0.684 |

Both trained models dramatically beat raw SigLIP — AUC +0.23, FITB accuracy +0.22 for Model A. Model A even edges past Vasileva et al.'s own published numbers (AUC 0.88, FITB 57.6%), though that comparison carries a real caveat: their number comes from a fully-trained type-conditioned network with a learned scoring function, while this phase uses a much simpler frozen-SigLIP-plus-small-MLP setup scored by plain mean pairwise similarity — a different, simpler scoring method outperforming on this particular formulation isn't the same claim as "a better model" in general. The comparison that matters most for this phase's question is unambiguous regardless: raw SigLIP vs. trained-on-clean-outfit-data is a landslide in favor of training. Consistent with phases 7-9's pattern, hard negatives (Model B) underperform random negatives (Model A) here too, though the gap is much smaller than on Amazon data.

## Step 5.2: transfer test on Amazon — the trained signal does NOT transfer, and transfers worse than Amazon's own noisy-data-trained model

| Configuration | Full HR@5 | Cross-type-only HR@5 |
|---|---|---|
| Raw SigLIP | 0.502 | 0.076 |
| Phase 8 Model A (Amazon-trained) | 0.441 | 0.078 |
| Phase 8 Model B (Amazon-trained) | 0.362 | 0.059 |
| **Phase 9 Model A (Polyvore-trained)** | **0.315** | **0.046** |
| **Phase 9 Model B (Polyvore-trained)** | **0.261** | **0.038** |

Sanity check passed (recomputed raw SigLIP matches the existing baseline exactly). Both Polyvore-trained models underperform not just raw SigLIP but **phase 8's own Amazon-trained models, on both views** — despite Polyvore training winning by a huge margin on its own domain. This is the central finding of this phase: a compatibility signal learned from clean, purpose-built data does not automatically generalize across image domains, and can transfer *worse* than a signal learned from noisier but domain-matched data.

## Step 6: qualitative check — explains why, and reveals a color-matching heuristic

`qualitative_examples/` has 8 Polyvore-domain grids and 8 Amazon-domain (transfer) grids, including phase 8's laundry-bag partial-success case.

**On Polyvore's own domain, the qualitative results are excellent and explain the AUC/FITB numbers directly.** A red ankle boot query retrieves, under Model A: a matching red wool coat, red/gold bangle bracelets, red suede boots, and a gold-and-garnet ring — a genuinely stylist-coordinated outfit, something raw SigLIP (which just returns other ankle boots) cannot produce by construction. A rhinestone choker query under Model B retrieves sequin heels, a lace bodysuit, and matching rhinestone jewelry — again a coherent "evening glam" assembly, not a same-item-type list.

**On the Amazon transfer test, results are inconsistent, and reveal what the model actually learned.** For the Tommy Hilfiger watch query, Phase 9 Model A's #4 result is a genuinely excellent complement (a watch winder/box) — proof the learned signal *can* transfer well for some queries. But Model B's #1 result for the same query is a canvas medic bag, unrelated. For the pink drawstring laundry bag (phase 8's partial-success case), **both Phase 9 models converge almost entirely on bright-pink items regardless of category** — a pink sandal, a pink cap, a pink tank top, pink sneakers, pink underwear, pink toddler socks. This is a real, identifiable pattern, not noise: **the model appears to have learned color-coordination as a dominant, sometimes overriding, proxy for "compatibility."** This makes sense given how Polyvore outfits are actually styled (the red-boot and rhinestone examples above are also substantially color/theme matches), and it works well within Polyvore's fashion-styling context — but a shared bright color between a laundry bag and a pair of sneakers isn't a meaningful compatibility signal in Amazon's product-catalog context, and the transfer results show this heuristic misfiring there.

## Interpretation

Two things are true at once, and neither contradicts the other:
1. **The single-variable hypothesis is confirmed**: data quality, not architecture, was the binding constraint in phases 7-8. The identical architecture, trained on clean purpose-built outfit data instead of noisy co-purchase data, produces a dramatically better compatibility signal — on the domain it was trained for.
2. **That signal is domain-specific, not domain-general.** SigLIP's frozen embedding space encodes both domain-general visual structure and domain-specific style/photography conventions. A small MLP trained hard on Polyvore's specific visual world (styled fashion photography, strong color-coordination patterns) appears to overfit to those domain-specific patterns rather than learning a domain-transferable notion of "compatibility" — and the color-matching heuristic it leans on is a plausible, identifiable mechanism for why, not just a black-box degradation.

## Recommendation

**Polyvore-trained compatibility is worth carrying forward for a Polyvore-domain or closely-matched-domain deployment, but should NOT replace or blend with phase 8's Amazon-trained model for Amazon product recommendations** — on Amazon's own domain, it underperforms even phase 8's noisier-data-trained model, let alone raw SigLIP. Concretely:
- **If a future phase serves recommendations within fashion/outfit-styled imagery** (Polyvore-like content, or a product catalog with comparable styled photography), this phase's Model A is the strongest compatibility signal found anywhere in this project (AUC 0.947, FITB 70.3%) and should be the starting point.
- **For Amazon product-catalog recommendations specifically**, phase 8's Amazon-trained Model A remains the best available compatibility signal (still below raw SigLIP on the aggregate metric, but the best of the trained options, per phase 8's own recommendation to use it as a secondary "goes well with" signal, not a primary ranker).
- **Domain adaptation, not just clean data, is the open problem** if a Polyvore-quality compatibility signal is ever wanted on Amazon imagery: e.g. fine-tuning a Polyvore-pretrained head on a small amount of Amazon-domain data, or an explicit color/style-invariance regularizer to counteract the color-matching heuristic identified here, rather than assuming clean training data alone transfers.

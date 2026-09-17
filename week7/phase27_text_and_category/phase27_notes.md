# Phase 27: Text and Category Inputs -- Honest Interpretation

## Headline

**Text helps, clearly and cleanly. Category conditioning hurts, badly, regardless of which of the two tested encoding methods was used.** This is a genuine, informative split result, not a wash -- the two additions the brief asked to isolate turned out to point in opposite directions, and the phase's whole design (two separate ablations, single-seed, validation-only decisions, test touched once) is exactly what was needed to tell them apart cleanly.

## Does text help? Yes, at every K, over both single-model references

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Week 6 image-only, single model | 0.1505 | 0.2740 | 0.3503 |
| **Phase 27 text_only** | **0.1656** | **0.2947** | **0.3733** |
| Week 6 image-only, 10-model ensemble | 0.1767 | 0.3054 | 0.3828 |

Adding SigLIP text embeddings (cascaded description/title/url_name, decision #1) to the same architecture, same hyperparameters, same training data, produces a real +10.0%/+7.6%/+6.6% relative gain over the image-only single model at K=10/30/50 -- comparable in size to phase 23/24's own tuning gain (+11.8%/+8.9%/+7.0% over phase 9's original), and *from a single seed*, no ensembling. It doesn't quite reach week 6's 10-model ensemble (within 2.5-6.3% relative of it, closest at K=50), which is a fair comparison to name plainly: one model with richer per-item input against ten models with plainer input isn't the same kind of investment, and this phase was never asked to close that specific gap. The honest takeaway is that text is a real, cheap, single-seed win, and stacking it with ensembling (not attempted here, explicitly deferred per the brief) is an obvious, well-motivated next step.

## Does category conditioning help? No -- it substantially hurts, in both tested forms

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 27 text_only | 0.1656 | 0.2947 | 0.3733 |
| Phase 27 text + category (learned table) | 0.1059 | 0.2059 | 0.2719 |
| Phase 27 text + category (SigLIP phrase) | 0.1090 | 0.2130 | 0.2793 |

Both category-conditioned variants land **below even week 6's original image-only baseline** (0.1505/0.2740/0.3503) at every K -- not a small regression, a large one: -36.0%/-30.1%/-27.2% relative against `text_only` for the learned table, -34.2%/-27.7%/-25.2% for the SigLIP phrase. This is not a case of "no effect" -- it's an active, sizable harm.

## Does the category-encoding method matter? No -- both fail almost identically

The brief's secondary question (learned table vs. reusing SigLIP's own encoder for the category phrase) has a clean answer: it barely matters which one is used. 0.1059 vs. 0.1090 at K=10, 0.2059 vs. 0.2130 at K=30, 0.2719 vs. 0.2793 at K=50 -- both variants' training curves (`training_log.md`) have the same shape and nearly the same magnitude throughout. This rules out "the learned table just needed more/better training" as the explanation, since the frozen, pre-trained SigLIP-phrase version fails the same way with zero training of its own category vectors. Whatever is going wrong is about *how category conditioning is wired into the architecture*, not about the specific vectors used.

## Ruled out first: is this an architectural failure (category signal never actually engaged)?

This was the user's own explicitly requested check, run before drawing any conclusion from the numbers above, specifically to avoid conflating "category conditioning doesn't help" with "the category signal never reached the network in a usable form." `category_embedding_check.md`: the learned 11-category table's mean pairwise cosine similarity is **0.0815** (range -0.194 to +0.267), nowhere near the 0.95 collapse threshold, and the structure is intuitively sensible -- `hats`/`sunglasses` are the most similar pair (both head-worn accessories), `shoes`/`tops` the most different. **The category signal clearly did engage the network and produced a differentiated, structured embedding table.** The negative result is real, not an artifact of a dead or collapsed input.

## The likely actual mechanism: a training/evaluation mismatch this project has seen before

`training_log.md` flags the tell: both category-conditioned variants have markedly *lower* training loss than `text_only` (~2.5-3.0 vs. ~3.7-4.7) while scoring *much worse* on held-out Recall@10, and the gap widens every epoch. That's the classic signature of a shortcut available during training that doesn't exist at evaluation time.

Here's the specific mechanism this points to. Training negatives (`sample_negatives` in `modal_app.py`, unchanged from phase 9/23/25/26) are drawn **uniformly from the entire 251,008-item catalog**, with no category restriction. Evaluation candidate pools, by contrast, are **strictly category-restricted** (up to 3,000 items, all the same category as the query). Without category conditioning, this asymmetry is harmless -- the network has no way to know or exploit a negative's category, since it never sees the target category at all. Once category conditioning is added, the anchor/query is explicitly told its target's category on every training step, while most of its R=8 random negatives are, by chance, from *different* categories (any category besides the target's has roughly 10/11 of the catalog's mass). This makes "is this candidate even the right category" a trivially easy, almost-free signal the network can lean on to cut training loss -- and it is a signal that evaporates completely at evaluation time, where every candidate in the pool already shares the query's category. The network appears to have spent real capacity on this shortcut rather than on the fine-grained within-category compatibility signal that both `text_only` and week 6's image-only baseline had to learn properly, and evaluation Recall pays the price.

This is worth naming as a direct parallel to something this project has already diagnosed once before: phase 14's OutfitTransformer reproduction failed for structurally the same reason (training negatives not matching the category-restricted evaluation task), fixed in phase 14b by restricting negative sampling to match the real task. The same fix is a well-motivated, cheap next experiment here (draw negatives from the anchor's own target category when category conditioning is active) -- **not attempted in this phase**, which was scoped to a first signal check, but flagged as the clear next step if category conditioning is revisited.

One secondary, smaller confound is also on the record and shouldn't be silently forgotten: `implementation_notes.md`'s decision #4 elaboration notes that the category-conditioned variants necessarily use a different query-pooling mechanism (pre-projection pooling + one head call) than `text_only` (post-projection pooling, matching week 6). Given the size of the regression here (-27% to -36% relative) and the much more specific, evidence-backed shortcut-learning explanation above, the pooling-order difference is very unlikely to be the primary driver -- but it means a clean, fully isolated re-test of category conditioning (fixing the negative-sampling gap AND controlling for pooling order) is the right way to settle this definitively, not something to build further conclusions on top of without that.

## What this suggests for the next step

Per the brief's own framing: report what the numbers say, don't shape them toward an expected outcome. They say two different things clearly:

1. **Text is a real, cheap win** and belongs in the model going forward. The natural next step is combining it with ensembling (phase 26's own mechanism, not attempted here) to see whether the two gains stack the way tuning and scale did in weeks 4-6.
2. **Category conditioning, as implemented here, is a net negative**, and adopting it as-is would be a regression, not a fairness improvement. It should NOT be folded into the model going forward without first fixing the diagnosed training/evaluation mismatch (category-aware negative sampling) and re-testing -- exactly the kind of bounded, diagnosed, one-variable-at-a-time follow-up this project's own methodology (phases 14/14b, 16-18b) has repeatedly used successfully.

## What was NOT done in this phase, per its own scope

No cross-attention, no set-encoder transformer, no backbone fine-tuning, no compatibility-prediction pre-training, no curriculum learning for negatives, no ensembling of the `text_only` result, and no attempt at the category-aware-negative-sampling fix diagnosed above -- all explicitly out of scope for this initial signal check, all real candidates for a follow-up phase.

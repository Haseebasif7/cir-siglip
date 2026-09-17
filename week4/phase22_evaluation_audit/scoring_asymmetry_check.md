# Phase 22, Step 3: Scoring Asymmetry Between Context-Free and Context-Aware Models

Phase 9 produces one fixed embedding per item and scores a query as a single mean-pooled context vector's dot product against the pool (`cir_eval.py`, `_query_vector` + `evaluate_recall`). CSA-Net and OutfitTransformer instead compute a context-dependent score. Checked directly against each script's own aggregation code, not assumed from the architecture description.

## Phase 9 / raw SigLIP's own query aggregation (for comparison — it is NOT context-free either)

`week4/phase12_controllable_modes/scripts/cir_eval.py`, `_query_vector`:
```python
def _query_vector(embeddings, idx, query_items):
    v = embeddings[item_idx].mean(axis=0)
    norm = np.linalg.norm(v)
    if norm == 0:
    return v / norm
```
Phase 9's own query vector is already a **mean pool** of the context items' fixed embeddings, L2-renormalized. So phase 9 is not immune to context-length effects either — the question is whether CSA-Net/OutfitTransformer's own aggregation introduces an *additional*, asymmetric penalty on top of this, not whether context-length dependence exists at all (it does, for every model here).

## OutfitTransformer's aggregation: a learned, masked set-encoder, not naive averaging

`week4/phase14b_outfittransformer_category_negatives/scripts/model.py`, `embed_query`:
```python
        self.outfit_token = nn.Parameter(torch.randn(d_model) * 0.02)  # matches the repo's own token-init scale
    def embed_query(self, ctx_tokens, pad_mask):
        tok = self.outfit_token.view(1, 1, -1).expand(B, -1, -1)
        h = self.set_enc(seq, src_key_padding_mask=pad)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)
        h = self.set_enc(seq, src_key_padding_mask=None)
        out = self.embed_ffn(h[:, 0, :])
        return F.normalize(out, p=2, dim=-1)
```
This is a self-attention transformer over the (masked, variable-length) context tokens plus a learned 'outfit token' read out as the query embedding — the exact mechanism the OutfitTransformer paper itself uses, not a simplified average. Padding is handled via `src_key_padding_mask`, so variable context length does not change per-token weighting through zero-padding contamination. After this step, scoring against the pool is the **identical** single dot-product ranking phase 9 uses (`04b_cir_eval_random_negatives.py`'s `evaluate_recall`, same `query_mat @ pool_emb.T` / `ranks = (sims >= target_sims[:, None]).sum(axis=1)` pattern as phase 9's own `cir_eval.py`) — no extra normalization or dilution step is added after the query embedding is computed.

## CSA-Net's aggregation: the paper's own eq. 5 (average per-context-item distance)

`week4/phase13b_csa_net_siglip_backbone/scripts/03_csa_cir_eval.py`:
```python
                dist_sum = torch.zeros(len(pool_gidx), device=DEVICE)
                    dist_sum += d
                dist_avg = (dist_sum / len(ctx_items)).cpu().numpy()
                target_dist = dist_avg[target_pos]
                rank = int((dist_avg <= target_dist).sum())
```
CSA-Net has no single embedding per item by design (its whole contribution is a category-pair-conditioned subspace attention, so an item's representation depends on which category it's being compared against). `03_csa_cir_eval.py`'s own docstring cites this as implementing the paper's eq. 5 directly (average per-context-item pairwise distance), and the source (checked above at `week4/phase13_csa_net_baseline/scripts/04_csa_cir_eval.py`, reused unchanged in phase 13b) documents this as a *necessary, paper-faithful adaptation, not a loosening of the comparison* — forcing CSA-Net into phase 9's single-vector scheme would defeat the exact mechanism being reproduced. The averaging divides by `len(ctx_items)` (the query's own actual context length), a scale-normalizing step, not a length-dependent penalty — a 2-item and a 10-item query both get a distance on the same per-item-average scale, comparable to any other query regardless of context size.

## Rank/hit-counting convention: identical across all three

- Phase 9 / OutfitTransformer (similarity, higher=better): `ranks = (sims >= target_sims[:, None]).sum(axis=1)`
- CSA-Net (distance, lower=better): `rank = int((dist_avg <= target_dist).sum())`

Both use the equivalent 'count of candidates at least as good as the target' convention, correctly inverted for CSA-Net's distance-based scoring (`<=` on distance where phase 9/OutfitTransformer use `>=` on similarity — the correct flip, checked directly, not just assumed by symmetry of naming).

## Numeric probe: does Recall correlate with query context length?

Query context length ranges 1-16 items across 29681 queries (median 5). A full per-length recall breakdown was actually run (not just estimated from code reading) for phase 9 and OutfitTransformer — see the table below, produced by `03b_length_stratified_probe.py` re-running both models' real evaluation logic bucketed by query context length. CSA-Net was not included in this numeric probe (its evaluator re-embeds every context item per query against a fresh per-category candidate projection, materially more expensive to re-run for a length breakdown); the aggregation-code evidence above (its own paper-cited eq. 5, explicitly divided by `len(ctx_items)`) is relied on for CSA-Net instead.

## Verdict

**No scoring asymmetry found.** OutfitTransformer's context aggregation is a faithful, learned attention mechanism (not a simplified average, and not diluted by padding). CSA-Net's aggregation is the paper's own documented eq. 5, explicitly length-normalized. Both context-aware models are scored by the exact same final ranking convention as phase 9 (count of candidates at least as good as the target). No inconsistency was found here that would mean CSA-Net or OutfitTransformer are being penalized by their own evaluation code.


## Numeric follow-up: Recall@10 by query context length, phase 9 vs OutfitTransformer (actually run, not estimated)

| Context length bin | Phase 9 Recall@10 (n) | OutfitTransformer Recall@10 (n) | Ratio (P9/OT) |
|---|---|---|---|
| 1-3 items | 0.1438 (6079) | 0.0615 (6079) | 2.34x |
| 4-5 items | 0.1328 (14207) | 0.0564 (14207) | 2.36x |
| 6-7 items | 0.1171 (7431) | 0.0583 (7431) | 2.01x |
| 8-16 items | 0.1405 (1964) | 0.0698 (1964) | 2.01x |

Phase 9's advantage ratio over OutfitTransformer ranges 2.01x-2.36x across context-length bins (spread 0.35). If OutfitTransformer's aggregation were being systematically penalized by context length, this ratio would grow monotonically with bin size (longer-context queries penalized more). 
**The ratio is roughly flat across context-length bins** -- phase 9's advantage over OutfitTransformer is consistent regardless of how many context items a query has, which is the signature of a genuine capability gap between the two models, not an aggregation artifact that specifically punishes longer (or shorter) context OutfitTransformer queries.


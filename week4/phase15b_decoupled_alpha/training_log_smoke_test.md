# Phase 15b: Smoke Test Report

## Dead-gradient check (untrained model, one real 96-outfit batch)

- `masks`: grad_norm=0.097133
- `proj.weight`: grad_norm=6.332321
- `proj.bias`: grad_norm=8.235970
- `attn_net.0.weight`: grad_norm=0.000231
- `attn_net.0.bias`: grad_norm=0.000247
- `attn_net.2.weight`: grad_norm=0.000639
- `attn_net.2.bias`: grad_norm=0.000789

**No dead gradients** -- every shared parameter (including `attn_net`'s new alpha-input weights, unchanged from phase 15) receives a nonzero gradient at init under the decoupled dual-forward-pass loss.

## Collapse check (with vs. without uniformity regularizer)

5 epochs, 2000 train outfits, 400 val outfits, weight_sub=9.5801 (reused from phase 15's calibration), decoupled dual-forward-pass loss (complement fixed at alpha=0, substitute fixed at alpha=1, every step).

- no_uniformity: mean pairwise cosine similarity = 0.7512
- with_uniformity: mean pairwise cosine similarity = 0.3942

Uniformity regularizer enabled from the start of the real run per phases 13b/14/15's proactive convention regardless of this check's outcome, unless it shows the regularizer is actively harmful (not expected).

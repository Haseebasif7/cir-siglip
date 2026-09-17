# Phase 25: Honest Interpretation -- Does Scale Help, and Where Does It Stop?

## The headline number

Scale produced a real, if modest, improvement over phase 23/24's fully-tuned 256-width baseline: test-benchmark Recall@10/30/50 = **0.1505/0.2740/0.3503**, up from 0.1473/0.2684/0.3442 (+2.2%/+2.1%/+1.8% relative), on top of phase 9's original 0.1317/0.2464/0.3216. The gain came entirely from **width alone** -- wider hidden layers (256 -> 1024) -- not from depth or a larger embedding dimension, both of which were tested as separate variables and both of which either hurt or added nothing.

## What actually moved the needle

**1. Width helps modestly, and only at the correct learning rate.** Doubling (512) and quadrupling (1024) the hidden layer beat the 256-width baseline (val Recall@10=0.1600) at every learning rate tested except one (width=512/lr=0.002). The best result, width=1024/lr=0.0005 (val Recall@10=0.1656), is a real +3.5% gain. But this gain is fragile to learning rate: at width=1024, lr=0.001 and lr=0.002 both land around 0.160 -- barely distinguishable from the baseline, and *below* width=512's own best result. Had the per-architecture learning-rate recheck this phase's brief specifically required not been done, width=1024 at phase 23/24's inherited lr=0.001 would have looked like a wash, and the real gain available at lr=0.0005 would have been missed entirely. This is exactly the risk the brief flagged in advance, and it materialized exactly as warned.

**2. Depth hurts badly, monotonically, and unambiguously.** Adding one extra hidden layer (depth=2) dropped validation Recall@10 from 0.1656 (depth=1) to a best of 0.1515 -- already a meaningful regression. Adding a second extra layer (depth=3) dropped it further to a best of 0.1274. Every learning rate tested at every depth made this worse, not better; no rescue was found. `overfitting_check.md` diagnoses why: deeper networks show a growing train/validation Recall@10 gap that is both larger and faster-growing than width alone produces (depth=3's gap reaches 0.44 by epoch 8, more than double width=1024's own 0.20 by epoch 5), while validation Recall@10 itself plateaus at a much lower ceiling almost immediately. This is overfitting -- excess capacity translating into memorization rather than generalization -- compounded by a secondary effect (deeper networks start from a less-differentiated embedding space at initialization and take longer to de-collapse), but the dominant story is capacity outrunning the ~1.37M-edge training set, not an optimization failure. Depth is not a lever worth pulling further for this mechanism at this data scale.

**3. A larger final embedding dimension does not help either.** Neither 256-d nor 512-d beat the 128-d winner (best alternative: out_dim=512/lr=0.0005 at 0.1646, still short of 0.1656). This is a much smaller effect than depth's sharp degradation -- more neutral than actively harmful -- but it's still a negative result: making the retrieval space itself bigger doesn't buy anything once the trunk capacity (width) is already right-sized.

## What the overfitting check showed, in one sentence

Every scale increase tested (width, depth, or embedding dimension) showed the identical qualitative pattern -- train Recall@10 climbs steadily while validation Recall@10 peaks within the first 1-2 epochs and then declines -- and the size of that train/validation gap grows monotonically with parameter count, confirming that this mechanism's practical ceiling on ~1.37M training edges is close to being reached already at a modest width increase, well before depth or embedding-dimension increases could plausibly help.

## What the sanity check showed

No degenerate collapse at any tested scale -- `mean_pairwise_cosine` starts moderate (0.14-0.50 depending on architecture depth) at initialization and drops to a small, stable value (0.004-0.017) within a few epochs in every single configuration tested, never approaching the near-1.0 value that would indicate collapse. Scale's real risk here is overfitting, not embedding degeneracy.

## Final, honest verdict

**Scale helps, but only a little, and only through one specific knob.** Width alone buys a genuine ~2% relative test-benchmark gain at the correct learning rate; depth is actively harmful; a larger embedding space is neutral-to-slightly-negative. This mechanism -- a small MLP projection head on frozen SigLIP features, trained on real Polyvore outfit co-occurrence -- appears close to saturated on this data at roughly the scale phase 23/24 already found, not meaningfully capacity-limited. This is a real, useful negative result for two of this phase's three variables (depth and embedding dimension), not a disappointment to round away: it tells any future session considering further scale increases (a bigger backbone, an ensemble, more data) that architecture size on top of this specific frozen-embedding mechanism has already returned nearly all the gain it's going to.

## Final reference configuration going forward

- **Architecture**: hidden_dims=[1024] (1 hidden layer, width 1024), out_dim=128 -- everything else (batch_size=256, weight_decay=0.0, tau=0.15, R=8) unchanged from phase 23/24.
- **Learning rate**: 0.0005 (down from phase 23/24's 0.001 -- the tuned lr did NOT carry over unchanged at the new width, confirming the brief's own stated risk).
- **Test-benchmark result**: Recall@10/30/50 = 0.1505/0.2740/0.3503.
- **Checkpoint**: `week4/phase25_scale/models/final_scaled.pt`.

## Operational footnote

Two of the confirmatory-retrain attempts in this phase failed with `modal.exception.RemoteError` (a dropped/cancelled long-running remote call), most likely caused by the local machine sleeping mid-run and severing the connection to Modal partway through a multi-epoch training job. Resolved by wrapping the retrain command in `caffeinate -is` (prevent sleep for the call's duration) and increasing the Modal function's own timeout as a secondary safety margin; the third attempt succeeded cleanly. This is a minor operational detail about running long Modal jobs from a laptop that may sleep, not a finding about the model or data.

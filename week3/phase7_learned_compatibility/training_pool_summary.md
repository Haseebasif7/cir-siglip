# Phase 7: Training Pool Summary

## Data hygiene filtering (step 2)

- Starting pool (after step 1 sampling + download): 27970 products
- Corrupted-title filter (`\bvar\s|\bfunction\(|<script`, word-boundary-anchored after a plain `var ` substring match false-positived on brand name "Manyavar"): removed 187 products (0.67%)
- Duplicate-image filter (MD5 hash): 1081 duplicate groups found, 3064 products demoted to a single representative each
- **Final cleaned pool: 24719 products**

- also_buy list-total refs (not yet restricted to within-pool targets, same convention as every prior phase's raw ref counts): 1070864 raw (post title-filter, pre-dedup-remap) -> 978791 after remapping duplicate endpoints to their representative (1447 self-loops dropped, i.e. edges where both endpoints collapsed to the same representative after remapping). These still include references to products outside this pool (different category, filtered out, or never in the qualifying pool at all) -- step 4 restricts to edges where both endpoints are actually in the cleaned pool with a surviving embedding, which is the trainable positive-edge count reported there.

Example corrupted titles removed (up to 5 shown):

- `B00180R7RA`: 'var aPageStart = (new Date()).getTime();\nvar ue_t0=ue_t0||+new Date();\n\nwindow.ue_ihb = (window.ue_i'
- `B001GG4BVQ`: 'var aPageStart = (new Date()).getTime();\nvar ue_t0=ue_t0||+new Date();\n\nwindow.ue_ihb = (window.ue_i'
- `B00296Z98Q`: 'var aPageStart = (new Date()).getTime();\nvar ue_t0=ue_t0||+new Date();\n\nwindow.ue_ihb = (window.ue_i'
- `B001C7FI8O`: 'var aPageStart = (new Date()).getTime();\nvar ue_t0=ue_t0||+new Date();\n\nwindow.ue_ihb = (window.ue_i'
- `B001P80F1U`: 'var aPageStart = (new Date()).getTime();\nvar ue_t0=ue_t0||+new Date();\n\nwindow.ue_ihb = (window.ue_i'


## Training pairs (step 4)

- Positive also_buy edges: 76293
- Train edges: 68664
- Val edges: 7629 (10.0%)
- Anchors with >=1 hard-negative candidate: 13677
- Anchors with 0 hard-negative candidates (all too-similar/excluded): 0

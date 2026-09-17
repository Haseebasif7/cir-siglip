# Visual Similarity-Based Recommender Using Deep Learning

Mitacs GRI Project — Haseeb Asif

## Project Description

A visual similarity-based recommender system that enhances product discovery by
leveraging deep learning on product images. Explores how visual features (style,
color, shape, design) can recommend similar or complementary products in
ecommerce settings using CNNs and multimodal models to extract image embeddings
and build an efficient similarity search framework.

The work converged on complementary item retrieval (CIR): given part of an outfit and a target
category, rank the catalog items of that category so that the item actually worn with the outfit
ranks highly. The later phases compare three retrieval mechanisms over one shared frozen
vision-language backbone under matched conditions.

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How the work is organised

The project runs as a sequence of numbered phases. Each phase is self-contained: its own folder,
its own scripts, its own outputs, and a notes file written while the work was running rather than
reconstructed afterwards. Phases are never rewritten after the fact, so a phase whose conclusion
was later overturned still carries its original notes, with the correction recorded in the phase
that overturned it.

| Folder | Contents |
| --- | --- |
| `week1/` | Literature review and dataset selection (Amazon Review Data 2018, Clothing/Shoes/Jewelry). |
| `week2/` | Frozen embedding benchmarks (ResNet50, CLIP ViT-B/32, FashionCLIP, SigLIP), product cropping, popularity and long-tail analysis, DeepFashion cross-dataset check. |
| `week3/` | Substitute versus complement splits, learned compatibility heads, colour invariance, and the move to Polyvore Outfits with its released compatibility and fill-in-the-blank protocols. |
| `week4/` | CIR benchmark construction, two reproduced baselines (CSA-Net, OutfitTransformer), an evaluation audit, hyperparameter tuning and ensembling. |
| `week5/` | Long-tail and controllable-retrieval experiments. This line of work did not improve CIR and was closed out. |
| `week7/` | The matched-condition comparison: one frozen SigLIP backbone, three heads (mean-pooled projection, transformer set encoder, category-conditioned subspace attention), each given the same treatment. Phases 31 to 35. |
| `week8/` | Statistical and reporting prerequisites: paired-bootstrap equivalence testing, evaluation on Polyvore's officially released test files, inference cost, and parameter accounting. |

Inside a phase folder, `phaseN_notes.md` is the primary document: what was tried, what worked,
what did not, and the numbers actually produced. `results_table.md` and the other Markdown files
hold the supporting measurements. `scripts/` holds the code that produced them.

## What is in this repository, and what is not

Tracked: all experiment code, the as-built notes and result tables for every phase, figures, run
logs, and the trained weights for the final systems.

Not tracked: the datasets and the intermediate artifacts derived from them. That is roughly 24 GB
of images, archives and embedding matrices, all of it either a public download or an output of the
scripts here. Specifically, every `data/` and `embeddings/` directory, and all `.npz`, `.tar`,
`.zip` and image files, are excluded.

One exception is kept because it is small and the reported numbers are computed directly from it:
`week8/phase36_paper_prerequisites/data/` holds the per-query rank vectors and official-split
scores for every compared system.

## Reproducing the results

The source datasets are public. Amazon Review Data 2018 (McAuley Lab, UC San Diego) is used through
week 3; the Polyvore Outfits release of Vasileva et al. is used from week 3 onward, in its
nondisjoint split. Each phase's notes state which files it consumed.

The pipeline runs in two stages. Item features are extracted once with a frozen SigLIP base model
(patch 16, 224 px), image tower and text tower concatenated to 1536 dimensions and L2-normalized,
covering all 251,008 catalog items; the extraction scripts live in the phase that first needed
them. Heads then train on those cached features, which is why training is cheap: the backbone is
never fine-tuned anywhere in this project.

The final trained heads are committed, so the reported retrieval numbers can be reproduced without
retraining:

| System | Weights |
| --- | --- |
| Mean-pooled projection, 10 seeds | `week7/phase28_text_ensemble/models/` |
| Mean-pooled projection, single model | `week7/phase27_text_and_category/models/text_only.pt` |
| Transformer set encoder, 3 seeds | `week7/phase32_partial_ensemble_outfittransformer/models/` |
| Subspace attention, 3 seeds | `week7/phase34_csanet_random_negatives/models/` |

Evaluation entry points for the matched comparison are in
`week8/phase36_paper_prerequisites/scripts/`, which loads each checkpoint through its own scoring
function and reproduces each system's recorded aggregates before computing anything new.

## Naming

Where this project compares against published architectures, it compares against its own
implementation of the published *mechanism* under matched conditions, not against the published
system. Those runs are named accordingly in the notes, because they differ from the published
systems in backbone, depth, loss and training recipe. Published numbers, where quoted, are taken
from the original papers and are labelled as such.

## Author

Muhammad Haseeb Asif, under the supervision of Dr. Mahreen Nasir, Algoma University.

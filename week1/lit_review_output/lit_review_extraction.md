Notion Link : https://app.notion.com/p/Literature-Review-Visual-Similarity-Based-Recommendation-3a8a1afaaa9e80a085c6ed44bab7be45?source=copy_link

Content directly :

---

## 1. ResNet: Deep Residual Learning for Image Recognition

He et al., 2015 | arxiv.org/abs/1512.03385

Background/architecture paper, not a recommendation paper. Included because ResNet is commonly used as a generic image feature extractor inside recommendation systems.

- **Technique/model used:** A Convolutional Neural Network (CNN) built from "residual blocks." Normally, each layer in a deep network tries to learn a completely new transformation of its input. A residual block instead learns the *difference* (the "residual") between the input and the desired output, and adds that difference back onto the original input using a **skip connection**. In simple terms: instead of learning "turn X into Y," the layer learns "figure out what needs to be added to X to get Y," and a direct wire carries X forward so it doesn't have to be relearned. This trick is what let the authors train networks 100+ layers deep without them getting worse as they got deeper (a problem that plagued earlier deep networks).
- **How visual features are used for recommendation:** Not applicable. ResNet doesn't do recommendation, it's a classification network. Relevant only as a possible embedding generator for other techniques.
- **Dataset/category tested on:** ImageNet (general purpose image classification), not a product/recommendation dataset.
- **Metrics used to evaluate the system:** No recommendation metric here (not a recommendation paper), but the paper itself is evaluated with **Top 1 and Top 5 classification error rate** on ImageNet. This measures how often the correct class is the model's #1 guess, and how often it's anywhere in the top 5 guesses.
- **Key number/result:** Stripping ResNet50's final classification layer and keeping the layer before it (the average pooling layer) leaves a vector of **2048 numbers**. This is the "embedding" people mean when they say "we used ResNet features," and it's the vector size compared/searched over when using ResNet as an embedding method.
- **Stated weakness/limitation:** Not applicable. Architecture paper, no recommendation specific weaknesses claimed.

!image.png

!image.png

---

## 2. CLIP: Learning Transferable Visual Models From Natural Language Supervision

Radford et al., 2021 | arxiv.org/abs/2103.00020

Background paper. CLIP itself isn't a recommender, but it's the foundation that FashionCLIP and VLCLIP (papers 4 and 7 below) build on top of.

- **Technique/model used:** Two separate neural networks, one that encodes images and one that encodes text, trained together so that the embedding of an image and the embedding of its matching caption end up close together in the same vector space, while embeddings of *mismatched* image caption pairs are pushed apart. This training method is called a **contrastive loss**: the model is shown a batch of images and captions, and for each image, it has to correctly pick out its true caption from among all the others in the batch (and vice versa). Trained on 400 million image caption pairs scraped from the internet.
- **Zero shot capability:** Because images and text share the same embedding space, a new image can be classified into categories it was *never explicitly trained on*, just by comparing the image's embedding to text embeddings of category names/descriptions (e.g., comparing a photo's embedding to the embedding of the sentence "a photo of a dog"). No retraining needed, this is called **zero shot** classification. CLIP matched the accuracy of a fully supervised ResNet50 on ImageNet without ever training on ImageNet's 1.28 million labeled images.
- **Dataset/category tested on:** Not a product dataset. Evaluated zero shot across ImageNet and roughly 30 other general purpose image datasets.
- **Metrics used to evaluate the system:** **Zero shot Top 1 accuracy** (comparing image embeddings to text embeddings of class names, with no task specific training) across ImageNet and the ~30 other benchmark datasets, plus linear probe accuracy comparisons against other pretrained models.
- **Key number/result:** Same accuracy as supervised ResNet50 on ImageNet, achieved with zero task specific training data.
- **Stated weakness/limitation:** Not applicable. Not a recommendation paper.

Loss is calculated by averaging image-to-text loss and text-to-image loss . see https://medium.com/self-supervised-learning/understanding-siglip-the-more-efficient-vision-encoder-b0b5f4c6a233

!image.png

---

## 3. Image Based Recommendations on Styles and Substitutes

McAuley, Targett, Shi & Van Den Hengel, 2015 | arxiv.org/pdf/1506.04757

The original Amazon visual recommendation paper, the one that started the "use product photos to recommend related products" idea. More mathematically involved than the others.

- **Technique/model used:** Starts with generic CNN image features (pretrained, not their own trained network), then **learns a transformation on top of those features** specifically for the recommendation task.
- **How visual features are used for recommendation:** The important part they get right that a naive approach gets wrong. A naive approach would just measure plain cosine similarity or Euclidean distance between two raw CNN feature vectors and call that "visual similarity." Instead, they **learn a distance metric**, specifically a low rank **Mahalanobis distance**. Intuition: a plain distance treats every dimension of the CNN feature vector as equally important. But when the goal is "these two items look like they go together" (rather than "these two items look identical"), some dimensions of the raw features matter much more than others for that judgment. The learned distance re weights and rotates the raw feature space so that distance in the *new* space reflects "compatibility," not just raw visual resemblance. Mathematically: they learn a matrix `Y`, project each item's raw features `x` into a new K dimensional space via `s = xY` (the **"style space"**), then measure plain Euclidean distance *between the projected points* `s_i` and `s_j`. So "style space" means a learned, lower dimensional space where nearby points represent items that go well together, not necessarily items that look alike.
- **Dataset/category tested on:** Amazon product copurchase data across many categories: Books, Clothing/Shoes/Jewelry, Electronics, Home & Kitchen, and 7 others.
- **Metrics used to evaluate the system:** **Accuracy on a link prediction task**: given a pair of items, predict whether Amazon's "also bought" or "also viewed" data actually links them (random guessing = 50% baseline). Also reports **AUC (Area Under the ROC Curve)**, summarizing how well the model ranks true links above unrelated pairs across all thresholds, not just one.
- **Key number/result:** On the Books category with a style space dimension of K=100, the method reached **71.2% accuracy**, compared to **66.5%** for a simpler weighted nearest neighbor baseline. A meaningful improvement from learning the distance metric instead of using a fixed one.
- **Stated weakness/limitation:** The authors admit that a lot of real "also bought" relationships in the data **aren't actually driven by visual appearance at all** (e.g., people buy a phone and a phone case together, related, but not because they look alike). Since the model only has access to appearance, this mismatch introduces noise it can't fully account for.

---

## 4 VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback

He & McAuley, 2016 | arxiv.org/abs/1510.01784

Extends matrix factorization with a visual factor learned from product images, giving items a head start from their photo even before any user interaction. The cleanest example of visual features and collaborative filtering fused into one scoring function. Most modern visual recommendation work is a fancier version of this same skeleton.

- **Technique/model used:** Every product image goes through a pretrained CNN (AlexNet style, off the shelf) to get a 4096 dimensional feature vector `fᵢ`. A learned embedding matrix `E` projects this down to a small "visual factor" `θᵢ = E·fᵢ` (10 to 20 dims), with a matching user vector `θᵤ`.
- **How visual features are used for recommendation:** Extends the standard MF prediction equation with two new terms: `x̂ᵤᵢ = α + βᵤ + βᵢ + γᵤᵀγᵢ + θᵤᵀθᵢ + β'ᵀfᵢ`. `θᵤᵀθᵢ` is visual compatibility (does this user's taste match this item's style), and `β'ᵀfᵢ` is a visual bias (does the item just look appealing to everyone). Trained with the same BPR pairwise ranking loss, just with more parameters learned jointly.
- **Dataset/category tested on:** Amazon Women, Amazon Men, Amazon Phones, and Tradesy.com (a clothing bartering site).
- **Metrics used to evaluate the system:** AUC, on the full item set and separately on cold start items (new items with little or no interaction history).
- **Key number/result:** Beats BPR-MF (non visual baseline) by roughly 5 to 10% AUC overall, growing to more than 28% average improvement in cold start scenarios, where the visual factor carries most of the signal.
- **Stated weakness/limitation:** Visual features come from a frozen, pretrained CNN never fine tuned for the recommendation task, so the model only learns a linear projection on top of fixed features rather than adapting the visual representation itself, unlike later fine tuned approaches (FashionCLIP, VLCLIP).

## 5. FashionCLIP: Contrastive Language and Vision Learning of General Fashion Concepts

Chia et al., 2022 | arxiv.org/pdf/2204.03972

Takes CLIP (paper 2) and **fine tunes it specifically on fashion product data**.

- **Technique/model used:** Same architecture as CLIP (paired image encoder + text encoder trained with contrastive loss), but starting from CLIP's pretrained weights and continuing training on fashion specific image text pairs, so the embedding space becomes much better at fashion specific concepts (fabric, cut, style names, etc.) that generic CLIP wasn't exposed to much.
- **How visual features are used for recommendation:** Products and text queries are embedded into the same shared space, so retrieval works by embedding a query (text or image) and finding the nearest product embeddings, essentially similarity search, plus zero shot classification of products into categories via text prompts.
- **Dataset/category tested on:** Farfetch's product catalog: over 700,000 image text pairs, spanning 3,000+ brands and 188 fine grained categories.
- **Metrics used to evaluate the system:** **HITS@5**: measures, for a given query, whether the correct/relevant item appears somewhere in the top 5 results returned. A HITS@5 of 0.61 means the correct item showed up in the top 5 results 61% of the time.
- **Key number/result:** On the held out test set, fine tuned **FashionCLIP scored 0.61 HITS@5**, versus **plain (not fine tuned) CLIP at only 0.22 HITS@5**. Roughly a 3x improvement from domain specific fine tuning, a strong argument for fine tuning on a target domain rather than using an off the shelf embedding model.
- **Stated weakness/limitation:** The expected limitation here (struggling with non fashion concepts) does **not actually hold up** in the paper. It instead shows FashionCLIP still succeeding at retrieving some non fashion visual concepts (e.g., tigers, stylized cat prints on clothing) despite being fine tuned narrowly on fashion. Needs verification against the full paper before citing this point directly.

!image.png

---

## 6. VisNet: Deep Learning Based Large Scale Visual Recommendation and Search for Ecommerce

Shankar et al., 2017 (Flipkart) | arxiv.org/pdf/1703.02344

A production/deployment focused paper from Flipkart. Less about a novel model, more about making visual search work at real scale.

- **Technique/model used:** A CNN based on VGG16, modified with extra parallel shallow convolution layers, trained using **triplet based deep ranking**. Triplet training means: for every training example, the model sees an "anchor" image, a "positive" image (similar to the anchor), and a "negative" image (dissimilar), and is pushed, via a hinge loss, to place the anchor's embedding closer to the positive than to the negative. This directly optimizes for "similar items end up close together in embedding space," rather than optimizing for classification accuracy the way ResNet does.
- **How visual features are used for recommendation:** Once embeddings are learned, **Euclidean distance between embeddings** is used to do k nearest neighbor (kNN) retrieval: find the k catalog items whose embeddings are closest to a query image's embedding. The embeddings capture similarity across **"semantic granularities"**, meaning the notion of "similar" operates at multiple zoom levels simultaneously: coarse category level similarity (a T shirt is more similar to another T shirt than to a shoe) down to fine grained visual detail (stripe width, exact color shade, pattern type).
- **Dataset/category tested on:** The public **Exact Street2Shop** dataset (20,000 real world "street" photos matched against 400,000 catalog/shop images) plus an internal Flipkart Fashion dataset. Category focus: clothing (T shirts, shirts, tops, dresses).
- **Metrics used to evaluate the system:** **Recall@20** on Street2Shop (of the top 20 retrieved items, what fraction of true matches were found); **triplet prediction accuracy** on the internal dataset (did the model correctly rank the "positive" image closer than the "negative" one); **human evaluation ratings** of result quality; and the business metric **conversion rate (CVR)**.
- **Key number/result:** At deployment scale, the system serves **50 million products** and handles **2,000 search queries per second**. On the internal dataset, triplet accuracy reached **97.38%**, and human evaluators rated **97% of results as "Excellent."** The business metric they highlight is **conversion rate**, not retrieval accuracy: visual recommendations drove a **26% conversion rate**, compared to only **8 to 10%** for other (non visual) recommendation modules on the site. They care most about whether visual recs actually lead to a purchase, not just whether the retrieved images look similar.
- **Stated weakness/limitation:** Three explicit weaknesses: (1) the object localization step (finding *where* the product is in a photo before embedding it) only reaches **68.2% mAP** (mean average precision), localization itself is a meaningfully imperfect step; (2) switching to approximate nearest neighbor search (LSH, used to make search fast at scale) causes **more than a 10% drop in result quality** compared to exact search; (3) the system needs about **one week** to "bootstrap" (build up) its index from scratch, an operational constraint.

---

## 7. Improving Visual Recommendation on Ecommerce Platforms Using Vision Language Models

Mercari, 2025 | arxiv.org/html/2510.13359

The closest match to a direct CNN vs VLM comparison: a head to head test of a classic CNN embedding vs a modern vision language model (VLM) embedding for ecommerce visual search.

- **Technique/model used:** **SigLIP**, a vision language model similar to CLIP, but trained with a **sigmoid based contrastive loss** instead of CLIP's softmax based one (a training detail that makes it more efficient at scale). Uses a ViT B/16 image encoder, fine tuned on Mercari's own marketplace data.
- **How visual features are used for recommendation:** Product images are converted into embeddings, stored in an index, and searched using **approximate nearest neighbor (ANN) search**: a fast, scalable way to find the closest matching embeddings without comparing against every single item in the catalog one by one. Retrieves visually similar listings (by color, shape, pattern).
- **Dataset/category tested on:** 1 million product image title pairs from Mercari's marketplace, collected April to July 2024, spanning all categories (Mercari is a general consumer to consumer marketplace, not fashion only).
- **Metrics used to evaluate the system:** **nDCG@5** (a ranking quality metric that rewards putting the most relevant results near the top, not just anywhere in the top 5), **Precision@1** and **Precision@3**, plus real business metrics: **click through rate (CTR)** and **conversion rate (CVR)**.
- **Key number/result:** A direct CNN vs VLM comparison: the CNN baseline was **MobileNetV2** (pretrained on ImageNet), and it was replaced with fine tuned **SigLIP** (the VLM). Offline results: SigLIP reached **0.662 nDCG@5**vs. MobileNetV2's **0.607**, a **9.1% relative improvement**. Precision@1 improved even more: **0.412 vs. 0.356**, a **15.7% relative gain**. The VLM (SigLIP) clearly beat the CNN (MobileNetV2) on offline retrieval quality.
- **Stated weakness/limitation:** No dramatic failure case reported, but an important practical trade off: when the embeddings were compressed from 768 dimensions down to 128 dimensions (using PCA, to make deployment cheaper/faster), retrieval quality dropped. nDCG@5 fell by **2.3%**. Useful reference point for embedding compression trade offs.

---

## 8. VLCLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM Augmented CLIP Embeddings

2025 | arxiv.org/pdf/2507.17080

Stacks two extra ideas on top of CLIP: **object level visual grounding** and **LLM enriched text**, then benchmarks against the other embedding methods in this reading list.

- **Technique/model used:** Fine tuned CLIP, combined with **Grounding DINO** (an object detection model) for visual grounding, plus an LLM used to clean up/enrich product text descriptions before encoding them.
- **How visual features are used for recommendation:** Two additions on top of plain CLIP: (1) **Visual grounding**: before an image is fed to CLIP's image encoder, Grounding DINO locates and crops the actual product region in the photo (removing background clutter), producing a cleaner, more product focused image for embedding. (2) **LLM augmented text**: product text descriptions (often messy or inconsistent in real catalogs) are rewritten/enriched by an LLM before being fed to CLIP's text encoder, producing more informative text embeddings. Both changes feed into the same CLIP style joint embedding space, just with higher quality inputs on both sides.
- **Dataset/category tested on:** Walmart.com product data across **Fashion** and **Home** categories (7 million products used for training; 10,000 Fashion + 10,000 Home products held out for evaluation). Also validated on Google Shopping and Walmart Art/Toys data.
- **Metrics used to evaluate the system:** **HITS@5** (same "is the right item in the top 5?" metric as FashionCLIP) and **MRR**: **Mean Reciprocal Rank**, which scores *how high* the correct item ranks (1st place scores higher than 5th place), rather than just whether it's in the top 5 at all. Online, also tracks **CTR, add to cart (ATC) rate, and GMV**.

| Model | Fashion HITS@5 | Fashion MRR | Home HITS@5 | Home MRR |
| --- | --- | --- | --- | --- |
| CLIP (plain) | 0.308 | 0.239 | 0.236 | 0.175 |
| GCL | 0.399 | 0.295 | 0.310 | 0.231 |
| FashionCLIP | 0.443 | 0.356 | 0.423 | 0.322 |
| **VLCLIP** | **0.676** | **0.525** | **0.669** | **0.510** |

VLCLIP clearly beats every other method on both categories and both metrics. Roughly a 50%+ relative improvement over FashionCLIP, and more than double plain CLIP. This also held up in a live online A/B test: **+18.6% click through rate, +15.5% add to cart rate, +4% GMV** (gross merchandise value, i.e. actual sales dollars) versus the previous production system.

- **Stated weakness/limitation (Not limitation for VLCLIP):** The paper frames its own contributions around three specific weaknesses of plain CLIP that it's trying to fix: (1) **weak object level alignment**: CLIP's embeddings are global (whole image), so they can miss specific product attributes if the product is small in the frame or surrounded by clutter; (2) **ambiguous textual representations**: real product catalogs often have inconsistent, incomplete, or noisy text descriptions, which hurts the text side of the embedding; (3) **domain mismatch**: CLIP was pretrained on general internet images/text, not ecommerce data specifically, so its embedding space isn't naturally tuned for product catalog nuances. These three points are essentially "why VLCLIP was needed" rather than a limitation of VLCLIP itself.

---

## 9. TGQFormer: Text Guided Visual Representation Learning for Robust Multimodal Ecommerce Recommendation

Guo, Ma, Zhang, Yang, Zang, Ding, Gong & Han, 2026 (KDD 2026) | arxiv.org/abs/2605.17366

Directly relevant for two reasons: (1) it tackles the problem of messy, real world ecommerce photos (banners, promo overlays, background clutter) rather than clean catalog images, and (2) for reproducibility, the authors **specifically rerun their method on the Amazon Clothing, Shoes and Jewelry dataset**: the same dataset/category used for the planned experiment here.

- **Technique/model used:** **TGQFormer (Text Guided QFormer)**. A plain "QFormer" is a small module that sits between a frozen image encoder (here, CLIP) and a large language model (LLM), using a set of learnable "query" vectors that cross attend to the image's visual tokens and compress them into a handful of embeddings the LLM can consume. TGQFormer's twist is splitting those queries into **two streams**: (a) "metadata anchored" queries, initialized from the product's text metadata (title, brand, category) so they know *what to look for* in the image, and (b) "exploratory" queries, which scan the image more freely for anything the metadata might have missed. A **dual gated modulation** mechanism then adaptively decides, per item, how much to trust each stream (e.g., down weighting the exploratory stream when an image is cluttered with promotional banners), and a regularizer discourages the two streams from learning redundant information.
- **How visual features are used for recommendation:** The frozen CLIP image encoder produces raw visual tokens → TGQFormer's hybrid queries compress these (guided by text metadata) into a small set of embeddings → these are fed into a pretrained LLM alongside the product's metadata embeddings → the LLM's final hidden state at a designated position becomes a single 256 dimensional item embedding. Retrieval is then just **cosine similarity search** between item embeddings, same general idea as the other papers here, but with a much more elaborate embedding generation pipeline upstream.
- **Dataset/category tested on:** Two datasets: (1) a large internal ecommerce platform's checkout/add to cart logs (3.5M training pairs, 67K test pairs); and (2) for reproducibility, the **Amazon Clothing, Shoes and Jewelry subset**(2018 Amazon product metadata), using the `also_bought`, `bought_together`, and `also_viewed` fields to define which items count as "true matches," the same style of ground truth McAuley et al. (paper 3) used.
- **Metrics used to evaluate the system:** **Hit Rate@K (H@K)** for K = 1, 5, 10, 20, 50, 100, evaluated as **full pool retrieval** (the correct item has to be found among *all* candidate items, not a small shortlist, a harder setting than it sounds). No NDCG or MRR reported in this paper.
- **Key number/result:** The full model reaches **H@100 = 69.13%**, beating the strongest prior connector baseline (NoteLLM2) at **65.19%**, a **+3.94 point absolute / 6.04% relative improvement**. It also beat a much bigger end to end multimodal LLM baseline (Qwen3 VL 4B Instruct, 65.68% H@100) while using **8 to 20x less compute** (35 GFLOPs vs. 592 GFLOPs). A smaller, purpose built module matched or beat a much larger general purpose model at a fraction of the cost. The ablation table shows each piece adding value: exploratory only queries alone reach 65.19% H@100, adding metadata anchoring pushes it to 67.45%, combining both (hybrid) reaches 68.11%, and adding the dual gated modulation gets to the final 68.90 to 69.13%.
- **Stated weakness/limitation:** No dedicated "Limitations" section, but two things stand out: (1) the method's gains are **much smaller on clean, object centric product photos** (like the Amazon benchmark) than on cluttered real world listing photos. The whole point of TGQFormer is fixing noisy images, so on a clean dataset like Amazon Clothing/Shoes/Jewelry, the benefit shrinks; (2) the method **depends on having structured text metadata** (title, brand, category hierarchy) to anchor the queries. Unclear how it would perform on items with sparse or missing metadata.

---
# Dataset Summary, Week 1

## Dataset
Amazon Review Data 2018 (McAuley Lab, UCSD)

## Link
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Direct metadata file: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz

## Category selected
Clothing, Shoes and Jewelry (2,685,059 products in the metadata file)

## Why this dataset

Amazon data keeps the project consistent with the visual similarity/fashion focus established through the literature review, and matches how prior work in the lit review used Amazon data:

- McAuley, Targett, Shi & Van Den Hengel (2015), "Image Based Recommendations on Styles and Substitutes", the original visual recommendation paper built on Amazon co purchase data across multiple categories including Clothing/Shoes/Jewelry.
- He & McAuley (2016), VBPR, tested visual factor based recommendation on Amazon Women, Men, and Phones categories, the same style of visually oriented Amazon subset.
- Guo et al. (2026), TGQFormer, specifically re ran their method on the Amazon Clothing, Shoes and Jewelry subset for reproducibility, using it as a standard benchmark category for this kind of experiment.

Using a category these papers used means results can be sanity checked against prior work rather than being an isolated test.

The 2018 version specifically (rather than the newer 2023 version) is used here because it includes `also_buy` and `also_viewed` relatedness fields needed for retrieval evaluation. The 2023 version dropped these down to a single sparser `bought_together` field.

## Major metadata fields

| Field | What it contains |
|---|---|
| `title` | Product name |
| `imageURL` / `imageURLHighRes` | Product photo URLs. Core input for embedding extraction |
| `price` | Price in USD at time of crawl |
| `feature` | Bullet point product features |
| `description` | Longer product description text |
| `categories` | Hierarchical category path for the product |
| `brand` | Brand name |
| `salesRank` | Sales rank information |
| `also_buy` | Items also bought with this one, used as a relatedness signal |
| `also_viewed` | Items also viewed alongside this one, used as a relatedness signal |
| `asin` | Product ID |

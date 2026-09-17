# Phase 8, Step 1: Fine-Grained Category Level Check

Every phase since 1b has used department-level categories (categories[1],
11 values: Women, Men, Boys, Girls, Baby, Novelty & More, Costumes & 
Accessories, Luggage & Travel Gear, Shoe/Jewelry/Watch Accessories, 
Traditional & Cultural Wear, Uniforms/Work/Safety). This phase needs a
finer level to distinguish real item types (a shirt vs. pants) within a
department, so heterogeneous-dyad filtering means something.

## Breadcrumb depth distribution (24,719 products, phase 7's cleaned pool)

| Breadcrumb length | Products |
|---|---|
| 2 | 257 |
| 3 | 1032 |
| 4 | 6689 |
| 5 | 5653 |
| 6 | 1133 |
| 7 | 1183 |
| 8 | 957 |
| 9 | 2164 |
| 10 | 2306 |
| 11 | 1502 |
| 12 | 1120 |
| 13 | 532 |
| 14 | 168 |
| 15 | 23 |

(Length includes the root 'Clothing, Shoes & Jewelry' at index 0 and 
department at index 1, so a length-3 breadcrumb has exactly one level 
beyond department; longer breadcrumbs increasingly trail off into 
product-feature bullet text rather than real category structure.)

## Candidate levels inspected

### Index 2: 164 distinct values, 257 products missing this depth

Top 15 by frequency: `Clothing` (5931), `Watches` (2620), `Jewelry` (2551), `Baby Girls` (1741), `Shoes` (1533), `Jewelry Accessories` (1141), `Shoe Care & Accessories` (1001), `Baby Boys` (992), `Accessories` (903), `Women` (854), `Backpacks` (794), `Men` (741), `Kids & Baby` (553), `Travel Accessories` (476), `Watch Accessories` (420)

### Index 3: 710 distinct values, 1289 products missing this depth

Top 15 by frequency: `Wrist Watches` (2412), `Novelty` (2038), `Clothing` (1723), `Accessories` (1063), `Jewelry Boxes & Organizers` (901), `Earrings` (875), `Costumes & Cosplay Apparel` (619), `Shoes` (475), `Shoelaces` (444), `Necklaces` (433), `Dresses` (428), `Bracelets` (422), `Kids' Backpacks` (396), `Athletic` (386), `Tops & Tees` (358)

### Index 4: 2850 distinct values, 7978 products missing this depth

Top 15 by frequency: `Men` (1141), `Costumes` (868), `Women` (644), `Imported` (472), `Jewelry Boxes` (430), `Clothing Sets` (349), `Drop & Dangle` (309), `Tees` (279), `Stud` (277), `Headwear` (248), `Pendants` (244), `Casual` (237), `Bodysuits` (223), `Footies & Rompers` (202), `Pajama Sets` (192)

## Complication found: breadcrumb semantics shift by department

A single fixed index does NOT consistently land on 'item type' across departments -- inspecting sample breadcrumbs per department shows why:

**Baby**:
  - ['Clothing, Shoes & Jewelry', 'Baby', 'Baby Boys', 'Clothing', 'Clothing Sets', 'Short Sets']
  - ['Clothing, Shoes & Jewelry', 'Baby', 'Baby Girls', 'Accessories', 'Hair Accessories', 'Imported']
  - ['Clothing, Shoes & Jewelry', 'Baby', 'Baby Boys', 'Clothing', 'Clothing Sets', 'Short Sets']
  - ['Clothing, Shoes & Jewelry', 'Baby', 'Baby Girls', 'Accessories', 'Hair Accessories', '100% Cotton']
**Boys**:
  - ['Clothing, Shoes & Jewelry', 'Boys', 'Shoes', 'Sneakers']
  - ['Clothing, Shoes & Jewelry', 'Boys', 'School Uniforms', 'Clothing', 'Tops']
  - ['Clothing, Shoes & Jewelry', 'Boys', 'Clothing', 'Clothing Sets', 'Pant Sets', '60% Cotton, 40% Polyester; 100% Polyester']
  - ['Clothing, Shoes & Jewelry', 'Boys', 'Clothing', 'Tops & Tees', 'Tees']
**Costumes & Accessories**:
  - ['Clothing, Shoes & Jewelry', 'Costumes & Accessories', 'Kids & Baby', 'Boys', 'Accessories']
  - ['Clothing, Shoes & Jewelry', 'Costumes & Accessories', 'Kids & Baby', 'Girls', 'Costumes', 'Includes: a pair of gorgeous blue gloves, an enchanting snowflake wand, a shimmering tiara, and a beautiful blue elegant snowflake necklace..']
  - ['Clothing, Shoes & Jewelry', 'Costumes & Accessories', 'Kids & Baby', 'Girls', 'Costumes', 'Fabric:Polyester']
  - ['Clothing, Shoes & Jewelry', 'Costumes & Accessories', 'Kids & Baby', 'Girls', 'Costumes', 'Imported']
**Girls**:
  - ['Clothing, Shoes & Jewelry', 'Girls', 'Shoes', 'Boots']
  - ['Clothing, Shoes & Jewelry', 'Girls', 'Shoes', 'Outdoor', 'Snow Boots', 'Nylon']
  - ['Clothing, Shoes & Jewelry', 'Girls', 'Accessories', 'Cold Weather', 'Gloves', 'Acrylic, ,']
  - ['Clothing, Shoes & Jewelry', 'Girls', 'Shoes', 'Boots']
**Luggage & Travel Gear**:
  - ['Clothing, Shoes & Jewelry', 'Luggage & Travel Gear', 'Umbrellas']
  - ['Clothing, Shoes & Jewelry', 'Luggage & Travel Gear', 'Messenger Bags', 'Leather', 'Highlights: genuine buffalo leather - Top quality - Classy, unisex vintage look', 'Features: thanks to the fine stitching and the excellent manufacturing, this genuine leather bag is a stylish and loyal companion for your everyday life.']
  - ['Clothing, Shoes & Jewelry', 'Luggage & Travel Gear', 'Luggage', '600-denier polyester', 'contains 50% recycled polyester', 'two self-fabric handles']
  - ['Clothing, Shoes & Jewelry', 'Luggage & Travel Gear', 'Backpacks', "Kids' Backpacks", 'Made of an easy to clean and very durable polyester cordura fabric with stunning embossed 3D logos and beautiful stamping', 'Features one large zippered compartment for binder, books and folders']
**Men**:
  - ['Clothing, Shoes & Jewelry', 'Men', 'Watches', 'Wrist Watches']
  - ['Clothing, Shoes & Jewelry', 'Men', 'Uniforms, Work & Safety', 'Clothing', 'Military', 'Pants']
  - ['Clothing, Shoes & Jewelry', 'Men', 'Jewelry', 'Rings']
  - ['Clothing, Shoes & Jewelry', 'Men', 'Clothing', 'Active', 'Active Vests']
**Novelty & More**:
  - ['Clothing, Shoes & Jewelry', 'Novelty & More', 'Clothing', 'Novelty', 'Men', 'Accessories']
  - ['Clothing, Shoes & Jewelry', 'Novelty & More', 'Clothing', 'Novelty', 'Men', 'Shirts']
  - ['Clothing, Shoes & Jewelry', 'Novelty & More', 'Clothing', 'Novelty', 'Men', 'Shirts']
  - ['Clothing, Shoes & Jewelry', 'Novelty & More', 'Clothing', 'Novelty', 'Men', 'Accessories']
**Shoe, Jewelry & Watch Accessories**:
  - ['Clothing, Shoes & Jewelry', 'Shoe, Jewelry & Watch Accessories', 'Jewelry Accessories', 'Jewelry Boxes & Organizers', 'Jewelry Trays']
  - ['Clothing, Shoes & Jewelry', 'Shoe, Jewelry & Watch Accessories', 'Watch Accessories', 'Cabinets & Cases']
  - ['Clothing, Shoes & Jewelry', 'Shoe, Jewelry & Watch Accessories', 'Jewelry Accessories', 'Jewelry Boxes & Organizers', 'Jewelry Boxes']
  - ['Clothing, Shoes & Jewelry', 'Shoe, Jewelry & Watch Accessories', 'Jewelry Accessories', 'Jewelry Boxes & Organizers', 'Jewelry Boxes']
**Traditional & Cultural Wear**:
  - ['Clothing, Shoes & Jewelry', 'Traditional & Cultural Wear', 'Asian', 'East Asian']
  - ['Clothing, Shoes & Jewelry', 'Traditional & Cultural Wear', 'Asian', 'South Asian', 'Chanderi Cotton', 'Embroidered']
  - ['Clothing, Shoes & Jewelry', 'Traditional & Cultural Wear', 'Middle Eastern', '100% Rayon', 'Imported', 'Fabric: Rayon (100% viscose) , Round neckline.']
  - ['Clothing, Shoes & Jewelry', 'Traditional & Cultural Wear', 'Middle Eastern', 'Lightweight and comfortable 1 piece Hijabs .', 'Pretty colors with rhinestones accents.', 'Just put on and be done .']
**Uniforms, Work & Safety**:
  - ['Clothing, Shoes & Jewelry', 'Uniforms, Work & Safety', 'Clothing', 'Food Service', 'Hats', '100% spun polyester']
  - ['Clothing, Shoes & Jewelry', 'Uniforms, Work & Safety', 'Clothing']
  - ['Clothing, Shoes & Jewelry', 'Uniforms, Work & Safety', 'Clothing', 'Medical']
  - ['Clothing, Shoes & Jewelry', 'Uniforms, Work & Safety', 'Clothing']
**Women**:
  - ['Clothing, Shoes & Jewelry', 'Women', 'Shoes', 'Boots', 'Ankle & Bootie', 'Leather']
  - ['Clothing, Shoes & Jewelry', 'Women', 'Clothing', 'Tops, Tees & Blouses', 'Blouses & Button-Down Shirts', 'Imported']
  - ['Clothing, Shoes & Jewelry', 'Women', 'Clothing', 'Socks & Hosiery', 'Tights', '25% polyester']
  - ['Clothing, Shoes & Jewelry', 'Women', 'Jewelry', 'Earrings', 'Hoop']

Concretely: Baby-department products put a gender subdivision ('Baby Girls'/'Baby Boys') at index 2, pushing the real item type ('Clothing Sets', 'Footies & Rompers', 'Hair Accessories') to index 4, not index 3. Novelty & More puts 'Clothing' at index 2, the vague bucket 'Novelty' at index 3, and ANOTHER demographic split ('Men'/'Women') at index 4 -- real type is even deeper. Costumes & Accessories similarly stacks 'Kids & Baby' then 'Boys'/'Girls' before reaching anything type-like. Traditional & Cultural Wear encodes *region* ('Asian', 'Middle Eastern') rather than garment type at all, and degenerates into free-text product description almost immediately (e.g. index 3 = 'Lightweight and comfortable 1 piece Hijabs.' for some products). Uniforms, Work & Safety terminates at length 3 for many products (no type info beyond 'Clothing' at all).

**Decision: use index 3 as the type level, with no per-department correction, and treat any product whose breadcrumb doesn't reach this depth as unknown type (excluded from heterogeneous-dyad determination in step 2, not guessed at).** This is a deliberately conservative choice: building a bespoke per-department index-correction table would reduce the noise described above, but it's out of scope for what this phase needs. The failure mode of NOT correcting for it is asymmetric and safe for this phase's purpose -- when index 3 is still a generic/demographic bucket rather than a true type (e.g. two different actual item types both showing 'Novelty'), the two products get scored as *same* type and their edge is excluded from the heterogeneous-dyad positive set. That undercounts genuine heterogeneous dyads in the affected departments (a real cost, addressed if needed by pool expansion in step 2), but it does NOT let same-type/near-duplicate pairs slip through mislabeled as heterogeneous -- which is the specific contamination phase 7 diagnosed and this phase exists to avoid. Index 3 clears the bar of 'genuinely fine-grained' overall (710 distinct values vs. 11 department buckets), even though it is not uniformly clean across every department.

## Result: 23430 of 24719 products have a known type at index 3 (94.8%); 1289 (5.2%) have breadcrumbs too shallow to reach it and are treated as unknown type.


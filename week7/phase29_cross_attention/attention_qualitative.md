# Phase 29: Attention Qualitative Check

Sample of 6 real validation queries (context length >= 3), using the single-seed (42) checkpoint from the step 3 gate. For each query, the attention distribution over context items is shown twice: conditioned on the TRUE target (what the model was actually trained on) and conditioned on a few random pool candidates (what it actually faces at evaluation time).

## Query 12677 (category: shoes, context length 5)

Context items: `gucci padlock small shoulder bag`; `patek philippe calatrava ref 2526`; `house harlow robyn round-frame acetate`; `Work Custom Skinny Jeans Product Fab new jeans in `; `saint laurent love-appliqu%C3%A9 cotton-blend mili`

True target: `Curtis 80 brown suede ankle boots by Saint Laurent`

- Conditioned on TRUE TARGET (`Curtis 80 brown suede ankle boots by Saint Laurent`): [gucci padlock small =0.21, patek philippe calat=0.08, house harlow robyn r=0.06, Work Custom Skinny J=0.39, saint laurent love-a=0.26] (entropy=1.42, normalized=0.88)
- Conditioned on random negative #1 (`wine red 39 suede spliced`): [gucci padlock small =0.55, patek philippe calat=0.05, house harlow robyn r=0.07, Work Custom Skinny J=0.11, saint laurent love-a=0.22] (entropy=1.24, normalized=0.77)
- Conditioned on random negative #2 (`Delicate lace extends the elegant look of an allur`): [gucci padlock small =0.51, patek philippe calat=0.03, house harlow robyn r=0.02, Work Custom Skinny J=0.31, saint laurent love-a=0.12] (entropy=1.16, normalized=0.72)
- Conditioned on random negative #3 (`These fluorescent pink Aquazzura 'Linda' barely th`): [gucci padlock small =0.45, patek philippe calat=0.06, house harlow robyn r=0.07, Work Custom Skinny J=0.33, saint laurent love-a=0.09] (entropy=1.30, normalized=0.81)

## Query 7734 (category: tops, context length 5)

Context items: `beige face pattern straw clutch`; `apricot peep toe sling back`; `shein sheinside pink frame metal`; `shein sheinside light blue bleached`; `shein sheinside contrast scallop trim`

True target: `Blue Florals Drop Shoulder T-shirt Shoulder(cm): X`

- Conditioned on TRUE TARGET (`Blue Florals Drop Shoulder T-shirt Shoulder(cm): X`): [beige face pattern s=0.15, apricot peep toe sli=0.22, shein sheinside pink=0.19, shein sheinside ligh=0.36, shein sheinside cont=0.09] (entropy=1.51, normalized=0.94)
- Conditioned on random negative #1 (`phase eight zita flocked top`): [beige face pattern s=0.28, apricot peep toe sli=0.14, shein sheinside pink=0.08, shein sheinside ligh=0.43, shein sheinside cont=0.07] (entropy=1.38, normalized=0.86)
- Conditioned on random negative #2 (`Turquoise blue metallic tie collar blouse from Del`): [beige face pattern s=0.25, apricot peep toe sli=0.19, shein sheinside pink=0.09, shein sheinside ligh=0.41, shein sheinside cont=0.05] (entropy=1.41, normalized=0.88)
- Conditioned on random negative #3 (`ALAND`): [beige face pattern s=0.04, apricot peep toe sli=0.53, shein sheinside pink=0.08, shein sheinside ligh=0.29, shein sheinside cont=0.07] (entropy=1.21, normalized=0.75)

## Query 181 (category: outerwear, context length 4)

Context items: `accessorize amani mini backpack`; `Whatever the day brings, Ugg's Sienna rain boots h`; `amazon.com jane stone pendant jewelry`; `river island black white check`

True target: `joules girls showerproof coat`

- Conditioned on TRUE TARGET (`joules girls showerproof coat`): [accessorize amani mi=0.12, Whatever the day bri=0.63, amazon.com jane ston=0.23, river island black w=0.01] (entropy=0.94, normalized=0.68)
- Conditioned on random negative #1 (`mother pearl mitchell oversized embellished`): [accessorize amani mi=0.62, Whatever the day bri=0.31, amazon.com jane ston=0.02, river island black w=0.05] (entropy=0.89, normalized=0.64)
- Conditioned on random negative #2 (`fenn wright manson florence cardigan`): [accessorize amani mi=0.26, Whatever the day bri=0.46, amazon.com jane ston=0.25, river island black w=0.03] (entropy=1.16, normalized=0.84)
- Conditioned on random negative #3 (`vanessa bruno open front crepe`): [accessorize amani mi=0.49, Whatever the day bri=0.31, amazon.com jane ston=0.03, river island black w=0.16] (entropy=1.12, normalized=0.81)

## Query 7064 (category: bags, context length 6)

Context items: `fashionable womens short boots with`; `kenneth jay lane womens double-sided`; `sterling silver openwork wolf band`; `Only US18.12, buy Stylish Scoop Neck Long Sleeve L`; `Balmain - Low-rise stretch cotton-denim biker jean`; `phillip lim leather biker jacket`

True target: `fashionable womens tote bag with`

- Conditioned on TRUE TARGET (`fashionable womens tote bag with`): [fashionable womens s=0.30, kenneth jay lane wom=0.05, sterling silver open=0.08, Only US18.12, buy St=0.24, Balmain - Low-rise s=0.17, phillip lim leather =0.16] (entropy=1.66, normalized=0.92)
- Conditioned on random negative #1 (`royce leather oversized airline ticket`): [fashionable womens s=0.00, kenneth jay lane wom=0.08, sterling silver open=0.71, Only US18.12, buy St=0.01, Balmain - Low-rise s=0.00, phillip lim leather =0.19] (entropy=0.85, normalized=0.48)
- Conditioned on random negative #2 (`alexander mcqueen heart clutch`): [fashionable womens s=0.21, kenneth jay lane wom=0.07, sterling silver open=0.07, Only US18.12, buy St=0.26, Balmain - Low-rise s=0.23, phillip lim leather =0.16] (entropy=1.69, normalized=0.94)
- Conditioned on random negative #3 (`marni colorblocked colubro bag`): [fashionable womens s=0.14, kenneth jay lane wom=0.04, sterling silver open=0.06, Only US18.12, buy St=0.35, Balmain - Low-rise s=0.23, phillip lim leather =0.17] (entropy=1.60, normalized=0.89)

## Query 17219 (category: bottoms, context length 5)

Context items: `punk pu leather chain studded`; `Buy your suede boots SAINT LAURENT on Vestiaire Co`; `black wool mix beret`; `Gently flared top in woven fabric with a V-neck at`; `biker jacket`

True target: `Womenswear Jeans CHEAP MONDAY (Material :  Rabbit `

- Conditioned on TRUE TARGET (`Womenswear Jeans CHEAP MONDAY (Material :  Rabbit `): [punk pu leather chai=0.12, Buy your suede boots=0.41, black wool mix beret=0.06, Gently flared top in=0.35, biker jacket=0.06] (entropy=1.32, normalized=0.82)
- Conditioned on random negative #1 (`Multi floral print gathered skirt with cutwork det`): [punk pu leather chai=0.24, Buy your suede boots=0.10, black wool mix beret=0.10, Gently flared top in=0.47, biker jacket=0.09] (entropy=1.37, normalized=0.85)
- Conditioned on random negative #2 (`Our Snowdrop tulle tutu skirt in Cherry Blossom is`): [punk pu leather chai=0.15, Buy your suede boots=0.11, black wool mix beret=0.03, Gently flared top in=0.68, biker jacket=0.03] (entropy=1.00, normalized=0.62)
- Conditioned on random negative #3 (`Logo detail Denim Solid colour Mid Rise Zip Five p`): [punk pu leather chai=0.05, Buy your suede boots=0.45, black wool mix beret=0.03, Gently flared top in=0.43, biker jacket=0.04] (entropy=1.11, normalized=0.69)

## Query 5519 (category: bags, context length 6)

Context items: `lojel groove 31.5 large spinner`; `anya hindmarch boom sticker`; `One of our sexiest strappy sandals yet, SANTI is c`; `covet monster ring set size`; `adriana degreas ruffled swimsuit`; `richard nicoll houndstooth jacquard sleeveless`

True target: `balenciaga blanket square medium floral`

- Conditioned on TRUE TARGET (`balenciaga blanket square medium floral`): [lojel groove 31.5 la=0.03, anya hindmarch boom =0.03, One of our sexiest s=0.11, covet monster ring s=0.12, adriana degreas ruff=0.46, richard nicoll hound=0.25] (entropy=1.43, normalized=0.80)
- Conditioned on random negative #1 (`pre-owned pink chanel caviar wallet`): [lojel groove 31.5 la=0.01, anya hindmarch boom =0.02, One of our sexiest s=0.61, covet monster ring s=0.02, adriana degreas ruff=0.12, richard nicoll hound=0.22] (entropy=1.10, normalized=0.61)
- Conditioned on random negative #2 (`shein sheinside zipper design pu`): [lojel groove 31.5 la=0.03, anya hindmarch boom =0.04, One of our sexiest s=0.37, covet monster ring s=0.07, adriana degreas ruff=0.25, richard nicoll hound=0.24] (entropy=1.49, normalized=0.83)
- Conditioned on random negative #3 (`maison margiela 11 clutch`): [lojel groove 31.5 la=0.02, anya hindmarch boom =0.02, One of our sexiest s=0.47, covet monster ring s=0.06, adriana degreas ruff=0.20, richard nicoll hound=0.24] (entropy=1.34, normalized=0.75)

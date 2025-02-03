# DivSuite


A Python framework for diversifying recommendation lists and evaluating them using metrics such as ILD, Recall, NDCG, and Hit Rate. This framework provides a modular design where different diversification algorithms (extending a common base) and embedding methods can be easily integrated and evaluated.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Requirements](#data-requirements)
  - [Top-K Recommendations](#top-k-recommendations)
  - [Item Mapping](#item-mapping)
  - [User Ground Truth](#user-ground-truth)


## Overview

The framework is designed to:

- **Diversify Recommendations:**  
  Apply various diversification algorithms (e.g., SYDiversifier, MMRDiversifier, etc.) that extend the `BaseDiversifier` class.

- **Generate Embeddings:**  
  Use open source Hugging Face models via the provided embedding methods to obtain item representations.

- **Evaluate Recommendations:**  
  Calculate metrics such as Intra-List Diversity (ILD), Recall, NDCG, and Hit Rate, allowing for flexible evaluation even when users have varying numbers of ground truth items.

## Project Structure
```
📦src
 ┣ 📂algorithms
 ┃ ┣ 📜base.py
 ┃ ┣ 📜bswap.py
 ┃ ┣ 📜clt.py
 ┃ ┣ 📜mmr.py
 ┃ ┣ 📜motley.py
 ┃ ┣ 📜msd.py
 ┃ ┣ 📜swap.py
 ┃ ┗ 📜sy.py
 ┣ 📂embedders
 ┃ ┣ 📜base_embedder.py
 ┃ ┣ 📜hf_embedder.py
 ┃ ┗ 📜ste_embedder.py
 ┣ 📂topk_data
 ┃ ┗ 📂ml100k
 ┃ ┃ ┣ 📜CMF_topk.pkl
 ┃ ┃ ┣ 📜target_item_id_mapping.csv
 ┃ ┃ ┗ 📜uid2positive_item.csv
 ┣ 📜README.md
 ┣ 📜config.py
 ┣ 📜environment.yml
 ┣ 📜logger.py
 ┣ 📜main.py
 ┗ 📜utils.py
```

## Data Requirements

### Top-K Recommendations

Place a `.pkl` file in the `topk_data/` folder that contains the top-K recommendations for each user. The expected format is a Python dictionary:
```python
{
    user_id: (list_of_titles, list_of_scores),
}
```
For example:
```json
{
    1: (['Scout, The (1994)'], [2.1710939407348633]),
    2: (['Scout, The (1994)'], [2.1220200061798096]),
    3: (["Devil's Own, The (1997)", "Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964)"], [1.93, 1.68]),
    ...
}
```

### Item Mapping

Include a CSV file named `target_item_id_mapping.csv` in the `topk_data/` folder. This file maps internal item IDs to their external representations (titles). The format should look like:

```csv
item_id,item
0,[PAD]
1,101 Dalmatians (1996)
2,12 Angry Men (1957)
3,187 (1997)
```

### User Ground Truth
Place a CSV file named `uid2positive_item.csv` in the `topk_data/` folder. This file contains the positive (ground truth) items for each user, with both user and item IDs in the internal format (as provided by recbole). For example:

```csv
user_id,positive_item
0,[]
1,"[514, 274, 1181, 816, 561, 436, 1076, 695, ...]"
2,"[193, 226, 803, 292, 1061, 102, 421, ...]"
3,"[416, 409, 539, 598]"
...
```
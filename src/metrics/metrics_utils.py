"""Utilities for computing and evaluating recommendation metrics."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

# Cache for storing loaded similarity scores
_SIMILARITY_SCORES_CACHE = {}


def load_similarity_scores(similarity_scores_path: str | None = None):
    """
    Load similarity scores from file and cache them for future use.

    Args:
        similarity_scores_path (str): Path to the pickle file containing precomputed similarity scores.

    Returns:
        dict: Dictionary containing similarity scores with integer keys extracted from tensors.
    """
    global _SIMILARITY_SCORES_CACHE

    # Return cached data if already loaded
    if similarity_scores_path in _SIMILARITY_SCORES_CACHE:
        return _SIMILARITY_SCORES_CACHE[similarity_scores_path]

    # Load data from file
    with open(similarity_scores_path, "rb") as f:
        sim_scores = pickle.load(f)

    # Convert tensor keys to integers for easier lookup
    lookup_dict = {}
    for (id1, id2), score in sim_scores.items():
        # Extract integers from tensors if needed
        key1 = id1.item() if hasattr(id1, "item") else id1
        key2 = id2.item() if hasattr(id2, "item") else id2
        lookup_dict[(key1, key2)] = score

    # Store in cache
    _SIMILARITY_SCORES_CACHE[similarity_scores_path] = lookup_dict

    return lookup_dict


def compute_average_ild_from_scores(
    topk_dict: dict,
    item_id_mapping: dict,
    similarity_scores_path: str | None = None,
    topk: int = 5,
) -> float:
    """
    Compute the average Intra-List Diversity (ILD) using precomputed pairwise similarity scores.

    ILD is defined as the average pairwise distance (1 - similarity) over all unique
    pairs within a user's top-k recommendation list.

    Args:
        topk_dict (dict): Dictionary with keys as user IDs and values as tuples of the form
                         (titles: [str], relevance_scores: [float]).
        item_id_mapping (dict): Dictionary mapping titles to item IDs.
        similarity_scores_path (str): Path to the pickle file containing precomputed similarity scores.
        topk (int, optional): The number of recommendations to consider per user. Defaults to 5.

    Returns:
        float: The average ILD value over all users.
    """
    # Load similarity scores (uses cache if available)
    lookup_dict = load_similarity_scores(similarity_scores_path)

    ild_list = []

    for _user, value in topk_dict.items():
        titles = value[0] if isinstance(value, tuple) and len(value) >= 1 else value
        titles = titles[: min(topk, len(titles))]

        # Convert titles to item IDs
        item_ids = [
            item_id_mapping.get(title) for title in titles if title in item_id_mapping
        ]

        n = len(item_ids)
        if n <= 1:
            ild_list.append(0.0)
            continue

        # Compute average distance (1 - similarity) for all pairs
        total_distance = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                id1, id2 = item_ids[i], item_ids[j]

                # Look up similarity score in both directions
                sim = lookup_dict.get((id1, id2))
                if sim is None:
                    sim = lookup_dict.get((id2, id1))

                # If similarity score exists, add distance to total
                if sim is not None:
                    total_distance += 1.0 - sim
                    pair_count += 1
                else:
                    # If no similarity score is found, use a default distance of 1.0 (no similarity)
                    total_distance += 1.0
                    pair_count += 1

        # Calculate average ILD for this user
        if pair_count > 0:
            ild = total_distance / pair_count
            ild_list.append(ild)
        else:
            ild_list.append(0.0)

    return np.mean(ild_list)


def compute_average_ild_batched(
    topk_dict: dict, embedder, topk: int = 5, precomputed_embeddings: dict | None = None
) -> float:
    """
    Compute the average Intra-List Diversity (ILD) over all users using batch embedding computation.

    ILD is defined as the average pairwise distance (1 - cosine similarity) over all unique
    pairs within a user's top-k recommendation list.

    Args:
        topk_dict (dict): Dictionary with keys as user IDs and values as tuples of the form
                         (titles: [str], relevance_scores: [float]).
        embedder: An instance with an encode_batch(list_of_titles) method.
        topk (int, optional): The number of recommendations to consider per user. Defaults to 5.
        precomputed_embeddings (dict, optional): A mapping from title (str) to its embedding (np.ndarray).
                                               If provided, these embeddings will be used instead of computing them.

    Returns:
        float: The average ILD value over all users.
    """
    # Step 1: Gather all top-k titles from all users
    all_titles = []
    user_indices = {}  # Map each user to a (start, end) tuple in the all_titles list
    current_index = 0

    for user, value in topk_dict.items():
        titles = value[0] if isinstance(value, tuple) and len(value) >= 1 else value
        current_titles = titles[: min(topk, len(titles))]
        all_titles.extend(current_titles)
        user_indices[user] = (current_index, current_index + len(current_titles))
        current_index += len(current_titles)

    # Step 2: Compute or retrieve embeddings for all titles in one batch
    all_embeddings = (
        np.array([precomputed_embeddings[title] for title in all_titles])
        if precomputed_embeddings is not None
        else embedder.encode_batch(all_titles)
    )

    # Step 3: For each user, extract their embeddings and compute ILD
    ild_list = []
    for user, (start, end) in user_indices.items():
        user_embeddings = all_embeddings[start:end]
        n = user_embeddings.shape[0]
        if n <= 1:
            ild_list.append(0.0)
            continue

        sim_matrix = cosine_similarity(user_embeddings)
        dist_matrix = 1 - sim_matrix
        iu = np.triu_indices(n, k=1)
        ild = np.mean(dist_matrix[iu])
        ild_list.append(ild)

    return np.mean(ild_list)


def evaluate_recommendation_metrics(relevance_lists: list, k: int) -> tuple:
    """
    Evaluate Precision@k, Recall@k, Hit@k, NDCG@k, and MRR@k for a set of recommendation lists.

    Args:
        relevance_lists (list): List of binary relevance vectors, where each vector represents
                               a user's recommendations (1 for relevant items, 0 for others).
        k (int): The cutoff rank at which to compute the metrics.

    Returns:
        tuple: (mean_precision, mean_recall, mean_hit, mean_ndcg, mean_mrr) where:
               - mean_precision: Average Precision@k over all users
               - mean_recall: Average Recall@k over all users
               - mean_hit: Average Hit@k over all users
               - mean_ndcg: Average NDCG@k over all users
               - mean_mrr: Average MRR@k over all users
    """
    metrics = []
    for rel in relevance_lists:
        rel_k = np.array(rel[:k])
        precision = np.sum(rel_k) / k
        recall = float(np.sum(rel_k) > 0)
        hit = int(np.sum(rel_k) > 0)
        scores = np.arange(len(rel_k), 0, -1)
        ndcg = ndcg_score([rel_k], [scores], k=k)

        # Calculate MRR for this user
        relevant_positions = np.where(rel_k == 1)[0]
        mrr = 1.0 / (relevant_positions[0] + 1) if len(relevant_positions) > 0 else 0.0

        metrics.append((precision, recall, hit, ndcg, mrr))

    return tuple(np.mean([m[i] for m in metrics]) for i in range(5))


def compute_average_category_ild_batched(
    rankings: dict, categories_dict: tuple, topk: int = 10
) -> float:
    """
    Compute average Intra-List Diversity based on genre vectors using cosine distance.

    Args:
        rankings (dict): Dictionary mapping user IDs to tuples of (titles, relevance_scores).
        categories_dict (tuple): Tuple of (title_to_categories dict, title_to_vector dict).
        topk (int, optional): Number of recommendations to consider per user. Defaults to 10.

    Returns:
        float: Average ILD value over all users, computed as:
               ILD = (2 / (len(list) * (len(list) - 1))) * sum(distance(i,j))
               where distance is cosine distance between genre vectors.
    """
    _, title_to_vector = categories_dict
    ild_list = []

    for _, (titles, _) in rankings.items():
        titles = titles[:topk]
        vectors = [v for t in titles if (v := title_to_vector.get(t)) is not None]

        if len(vectors) <= 1:
            ild_list.append(0.0)
            continue

        vectors_np = np.vstack(vectors)

        if not isinstance(vectors_np, np.ndarray):
            try:
                vectors_np = np.array(vectors, dtype=float)
            except Exception as e:
                print(f"Warning: Could not convert vectors to NumPy array. Error: {e}")
                ild_list.append(0.0)
                continue

        if vectors_np.ndim == 1:
            vectors_np = vectors_np.reshape(1, -1)

        if vectors_np.shape[0] <= 1:
            ild_list.append(0.0)
            continue

        pairwise_distances = pdist(vectors_np, metric="cosine")
        if len(pairwise_distances) > 0:
            ild = np.mean(pairwise_distances)
            ild_list.append(ild)
        else:
            ild_list.append(0.0)

    return np.mean(ild_list) if ild_list else 0.0


def precompute_title_embeddings(
    rankings: dict,
    embedder: Any = None,
    embedding_params: dict | None = None,
) -> dict:
    """
    Precompute embeddings for all unique titles (from each user's full recommendation list)
    and return a mapping from title to embedding.

    Parameters:
        rankings (dict): {user_id: (titles: [str], relevance_scores: [float]), ...}.
        embedder: An instance with an encode_batch(list_of_titles) method.
        embedding_params (dict): Embedding parameters (use_precomputed_embeddings: bool, precomputed_embeddings_path: str).
    Returns:
        dict: Mapping from title (str) to its embedding (np.ndarray).
    """
    if embedding_params["use_precomputed_embeddings"]:
        print(
            f"Loading precomputed embeddings from {embedding_params['precomputed_embeddings_path']}"
        )
        return load_precomputed_embeddings(
            embedding_params["precomputed_embeddings_path"], rankings
        )
    else:
        unique_titles = set()
        for _, (titles, _) in rankings.items():
            unique_titles.update(titles)  # use all items, not just top_k items.
        unique_titles = list(unique_titles)

        print(f"Computing embeddings for {len(unique_titles)} unique titles.")

        # Compute embeddings for all unique titles in one large batch.
        embeddings = np.array(embedder.encode_batch(unique_titles))

        # Return a mapping from title to embedding.
        return dict(zip(unique_titles, embeddings, strict=False))


def load_precomputed_embeddings(
    precomputed_embeddings_path: str, rankings: dict
) -> dict:
    """
    Load precomputed embeddings from file, filter based on titles in rankings,
    and return a mapping from title to embedding.

    Args:
        precomputed_embeddings_path (str): Path to the pickle file containing the DataFrame.
        rankings (dict): Rankings dictionary to extract relevant titles.

    Returns:
        dict: Mapping from title (str) to its embedding (np.ndarray).
    """
    with open(precomputed_embeddings_path, "rb") as f:
        embeddings_df = pickle.load(f)  # Assume this loads a pd.DataFrame

    # check if embeddings_df is a pd.DataFrame and have expected columns
    if not isinstance(embeddings_df, pd.DataFrame):
        raise ValueError("embeddings_df must be a pd.DataFrame")
    if "item" not in embeddings_df.columns or "embedding" not in embeddings_df.columns:
        raise ValueError("embeddings_df must have 'item' and 'embedding' columns")
    # Extract unique titles required from the rankings
    unique_titles_needed = set()
    for _, (titles, _) in rankings.items():
        unique_titles_needed.update(titles)

    # Filter the DataFrame to keep only the needed titles
    filtered_df = embeddings_df[embeddings_df["item"].isin(unique_titles_needed)]

    # Create the title-to-embedding dictionary
    title_to_embedding = {
        row["item"]: np.array(row["embedding"]) for _, row in filtered_df.iterrows()
    }

    # # print out dimension of embeddings
    # print(f"Embeddings dimension: {title_to_embedding[list(title_to_embedding.keys())[0]].shape}")

    return title_to_embedding

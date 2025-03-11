"""Utilities for computing and evaluating recommendation metrics."""

import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist


def compute_average_ild_batched(
    topk_dict: dict, embedder, topk: int = 5, precomputed_embeddings: dict = None
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
        if len(relevant_positions) > 0:
            mrr = 1.0 / (relevant_positions[0] + 1)
        else:
            mrr = 0.0

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

        vectors = np.vstack(vectors)
        pairwise_distances = pdist(
            vectors, metric="cosine"
        )  # TODO: check if this is correct
        ild = (2.0 / (len(vectors) * (len(vectors) - 1))) * np.sum(pairwise_distances)
        ild_list.append(ild)

    return np.mean(ild_list) if ild_list else 0.0

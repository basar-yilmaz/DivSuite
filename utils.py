from typing import Any
import numpy as np
from sklearn.metrics import ndcg_score, precision_score, recall_score
from numpy import floating
from sklearn.metrics.pairwise import cosine_similarity


def compute_pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """
    Given a 2D array of shape (N, dim) of embeddings,
    return an NxN matrix of pairwise cosine similarities.
    """

    # TODO this can be optimized by reducing space complexity
    return cosine_similarity(embeddings, embeddings)


def compute_ild_for_topk(recommendation_titles: list, embedder, top_k: int = 5) -> float:
    """
    Compute the Intra-List Diversity (ILD) for a single recommendation list.

    ILD is defined as the average pairwise distance (1 - similarity) across
    all unique pairs in the recommendation list.

    Parameters:
        recommendation_titles (list): List of titles (strings) for a user.
        embedder: An instance with an encode_batch(list_of_titles) method.
        top_k (int): The number of recommendations to consider.
    Returns:
        float: The ILD for the recommendation list.
    """
    if len(recommendation_titles) <= 1:
        return 0.0

    top_k = min(top_k, len(recommendation_titles))

    # Keep only the top k data points
    recommendation_titles = recommendation_titles[:top_k]

    # Encode the titles (assumes embedder.encode_batch returns a NumPy array)
    embeddings = embedder.encode_batch(recommendation_titles)
    # Compute the pairwise cosine similarity matrix.
    sim_matrix = compute_pairwise_cosine(embeddings)
    # Convert similarity to dissimilarity (distance)
    dist_matrix = 1 - sim_matrix

    n = len(recommendation_titles)
    # Sum upper triangle (excluding the diagonal) and average over number of pairs.
    total_distance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += dist_matrix[i, j]
            count += 1
    ild = total_distance / count if count > 0 else 0.0
    return ild


def compute_average_ild(topk_dict: dict, embedder, topk) -> floating[Any]:
    """
    Compute the average ILD metric over all users.

    Parameters:
        topk_dict (dict): Dictionary with keys as user ids and values as lists of titles.
        embedder: An instance with an encode_batch method used for computing embeddings.
        topk (int): The number of recommendations to consider.
    Returns:
        float: The average ILD value over all users.
    """
    ild_list = []
    for user, titles in topk_dict.items():
        user_ild = compute_ild_for_topk(titles[0], embedder, topk)
        ild_list.append(user_ild)
    avg_ild = np.mean(ild_list)
    return avg_ild

def compute_user_metrics(rec_titles, ground_truth_ids, title_to_id, k=10):
    """
    Compute recommendation metrics using scikit-learn.
    
    Parameters:
        rec_titles: List of recommended titles
        ground_truth_ids: List of ground truth item IDs
        title_to_id: Dictionary mapping titles to item IDs
        k: Number of items to consider
    Returns:
        tuple: (recall, ndcg, hit_rate, precision)
    """
    # Convert recommended titles to corresponding item IDs
    rec_ids = [title_to_id.get(title) for title in rec_titles][:k]
    rec_ids = [r for r in rec_ids if r is not None]
    
    # Create binary vectors for metrics calculation
    all_items = list(set(rec_ids + list(ground_truth_ids)))
    y_true = np.zeros(len(all_items))
    y_pred = np.zeros(len(all_items))
    
    # Fill binary vectors
    for idx, item in enumerate(all_items):
        if item in ground_truth_ids:
            y_true[idx] = 1
        if item in rec_ids:
            y_pred[idx] = 1
    
    # Calculate metrics using sklearn
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Calculate NDCG using sklearn
    relevance = np.array([[1 if r in ground_truth_ids else 0 for r in rec_ids]])
    ideal_relevance = np.sort(relevance, axis=1)[:, ::-1]
    ndcg = ndcg_score(ideal_relevance, relevance)
    
    # Calculate Hit Rate (still manual as sklearn doesn't have direct equivalent)
    hit_rate = 1 if any(r in ground_truth_ids for r in rec_ids) else 0
    
    return recall, ndcg, hit_rate, precision
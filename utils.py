import ast
import csv
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity


def load_data_and_convert(data_csv_path, mapping_csv_path):
    """
    Loads the data CSV file containing user interactions and converts the item IDs to titles
    using the provided mapping CSV. For each row, the function returns a flat list:
    [pos_item_title, neg_item_title1, neg_item_title2, ...]

    Parameters:
        data_csv_path (str): Path to the CSV file with columns "user_id", "pos_item", "neg_items".
        mapping_csv_path (str): Path to the CSV file with columns "item_id" and "item" (the title).

    Returns:
        list: A two-dimensional list, where each inner list is in the form:
              [pos_item_title, neg_item_title1, neg_item_title2, ...]
    """

    # Load the mapping CSV into a dictionary mapping IDs to titles.
    id_to_title = {}
    with open(mapping_csv_path, "r", encoding="utf-8") as map_file:
        reader = csv.DictReader(map_file)
        for row in reader:
            try:
                # Convert item_id to int (if possible) for consistency.
                key = int(row["item_id"])
            except ValueError:
                key = row["item_id"]
            id_to_title[key] = row["item"]

    # Process the data CSV file.
    results = []
    with open(data_csv_path, "r", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            # Convert the positive item ID to its title.
            try:
                pos_item_id = int(row["pos_item"])
            except ValueError:
                pos_item_id = row["pos_item"]
            pos_item_title = id_to_title.get(pos_item_id, str(pos_item_id))

            # Parse the neg_items field which is a string representation of a list.
            neg_items_str = row["neg_items"]
            try:
                neg_items_ids = ast.literal_eval(neg_items_str)
            except Exception as e:
                print(f"Error parsing neg_items: {neg_items_str}. Error: {e}")
                neg_items_ids = []

            # Convert each negative item ID to its title.
            neg_items_titles = []
            for item in neg_items_ids:
                try:
                    item_id = int(item)
                except ValueError:
                    item_id = item
                neg_items_titles.append(id_to_title.get(item_id, str(item_id)))

            # Combine the positive item title with the negative items titles into a single list.
            combined_list = [pos_item_title] + neg_items_titles
            results.append(combined_list)

    return results


def compute_average_ild_batched(topk_dict: dict, embedder, topk: int = 5) -> float:
    """
    Compute the average Intra-List Diversity (ILD) over all users using batch embedding computation.

    ILD is defined as the average pairwise distance (1 - cosine similarity) over all unique
    pairs within a user's top-k recommendation list.

    Parameters:
        topk_dict (dict): Dictionary with keys as user IDs and values as tuples of the form
                          (titles: [str], relevance_scores: [float]).
        embedder: An instance with an encode_batch(list_of_titles) method.
        topk (int): The number of recommendations to consider per user.

    Returns:
        float: The average ILD value over all users.
    """
    # Step 1: Gather all top-k titles from all users.
    all_titles = []
    user_indices = {}  # Map each user to a (start, end) tuple in the all_titles list.
    current_index = 0

    for user, value in topk_dict.items():
        # Check if the value is a tuple (titles, relevance_scores)
        if isinstance(value, tuple) and len(value) >= 1:
            titles = value[0]  # Extract the list of titles.
        else:
            titles = value  # In case it's just a list of titles.

        # Only take the top-k titles for this user.
        current_titles = titles[: min(topk, len(titles))]
        all_titles.extend(current_titles)
        user_indices[user] = (current_index, current_index + len(current_titles))
        current_index += len(current_titles)

    # Step 2: Compute embeddings for all titles in one batch.
    all_embeddings = embedder.encode_batch(
        all_titles
    )  # Expecting a NumPy array of shape (N, D)

    # Step 3: For each user, extract their embeddings and compute ILD.
    ild_list = []
    for user, (start, end) in user_indices.items():
        user_embeddings = all_embeddings[start:end]
        n = user_embeddings.shape[0]
        if n <= 1:
            ild_list.append(0.0)
            continue

        # Compute the pairwise cosine similarity matrix in a vectorized manner.
        sim_matrix = cosine_similarity(user_embeddings)
        # Convert similarity to dissimilarity (distance)
        dist_matrix = 1 - sim_matrix

        # Get the indices for the upper triangle (excluding the diagonal)
        iu = np.triu_indices(n, k=1)
        # Compute the average pairwise dissimilarity.
        ild = np.mean(dist_matrix[iu])
        ild_list.append(ild)

    avg_ild = np.mean(ild_list)
    return avg_ild


def evaluate_recommendation_metrics(relevance_lists, k):
    """
    Evaluate Precision@k, Recall@k, Hit@k, and NDCG@k for a set of recommendation lists.

    Each list in `relevance_lists` is a binary list (with one 1 and the rest 0's) ordered
    by the recommendation ranking. The 1 indicates the relevant item.

    Parameters:
      relevance_lists: List of lists. Each inner list is the binary relevance vector for a user.
      k: The cutoff rank at which to compute the metrics.

    Returns:
      mean_precision: Average Precision@k over all users.
      mean_recall: Average Recall@k over all users.
      mean_hit: Average Hit@k over all users.
      mean_ndcg: Average NDCG@k over all users.
    """
    precisions = []
    recalls = []
    hits = []
    ndcgs = []

    for rel in relevance_lists:
        # Only evaluate the top-k recommendations.
        rel_k = rel[:k]

        # Compute Precision@k:
        # With only one relevant item, if it appears in the top-k, precision is 1/k; otherwise 0.
        precision = np.sum(rel_k) / k

        # Compute Recall@k:
        # It is 1 if the relevant item is in the top-k, else 0.
        recall = 1.0 if np.sum(rel_k) > 0 else 0.0

        # Compute Hit@k:
        # Hit@k is 1 if at least one relevant item is in the top-k, else 0.
        hit = 1 if np.sum(rel_k) > 0 else 0

        # For NDCG, we use sklearn's ndcg_score.
        # ndcg_score expects 2D arrays: one for the true relevance and one for the scores.
        # We create a dummy score vector that reflects the ranking order (highest score for the first item).
        scores = np.arange(len(rel_k), 0, -1)  # For k=5, scores = [5, 4, 3, 2, 1]
        ndcg = ndcg_score([rel_k], [scores], k=k)

        precisions.append(precision)
        recalls.append(recall)
        hits.append(hit)
        ndcgs.append(ndcg)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_hit = np.mean(hits)
    mean_ndcg = np.mean(ndcgs)

    return mean_precision, mean_recall, mean_hit, mean_ndcg


def precompute_title_embeddings(rankings: dict, embedder) -> dict:
    """
    Precompute embeddings for all unique titles (from each user's full recommendation list)
    and return a mapping from title to embedding.

    Parameters:
        rankings (dict): {user_id: (titles: [str], relevance_scores: [float]), ...}.
        embedder: An instance with an encode_batch(list_of_titles) method.

    Returns:
        dict: Mapping from title (str) to its embedding (np.ndarray).
    """
    unique_titles = set()
    for user_id, (titles, _) in rankings.items():
        unique_titles.update(titles)  # use all items, not just top_k items.
    unique_titles = list(unique_titles)

    print(f"Computing embeddings for {len(unique_titles)} unique titles.")

    # Compute embeddings for all unique titles in one large batch.
    embeddings = embedder.encode_batch(unique_titles)

    # Return a mapping from title to embedding.
    return dict(zip(unique_titles, embeddings))

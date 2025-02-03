import pickle
import pandas as pd
import numpy as np
import ast

from algorithms.bswap import BSwapDiversifier
from algorithms.base import BaseDiversifier

from utils import compute_average_ild, compute_user_metrics


# ------------------------------------------
# 1. Diversification Pipeline
# ------------------------------------------
def run_diversification(
    rankings: dict, diversifier: BaseDiversifier, top_k: int = 10
) -> dict:
    """
    Run the diversification algorithm on recommendation titles and return
    the diversified output as a dictionary with the same structure as the input.

    Parameters:
        rankings (dict): {user_id: (titles: [str], relevance_scores: [float]), ...}.
        diversifier (BaseDiversifier): An instance of a diversification algorithm.
        top_k (int): Number of top items to select after diversification.

    Returns:
        dict: {user_id: (diversified_titles: [str], diversified_scores: [float]), ...}.
    """
    diversified_dict = {}

    for user_id, (titles, relevance_scores) in rankings.items():
        if len(titles) != len(relevance_scores):
            raise ValueError(
                f"User {user_id}: Number of titles and relevance scores do not match."
            )

        # Create an array with rows: [dummy_index, title, relevance_score]
        items = np.array(
            [[i, title, float(relevance_scores[i])] for i, title in enumerate(titles)],
            dtype=object,
        )

        # Run the diversification algorithm for this user.
        diversified_items = diversifier.diversify(items, top_k=top_k)

        # Extract diversified titles and scores (ignore the dummy index column)
        diversified_titles = diversified_items[:, 1].tolist()
        diversified_scores = [float(x) for x in diversified_items[:, 2]]

        diversified_dict[user_id] = (diversified_titles, diversified_scores)

    return diversified_dict


# ------------------------------------------
# 2. Data Loading Utilities
# ------------------------------------------
def load_item_mapping(csv_path: str):
    """
    Load the item mapping CSV and create two dictionaries:
      - id_to_title: mapping from item ID to item title.
      - title_to_id: reverse mapping from title to item ID.
    """
    mapping_df = pd.read_csv(csv_path)  # Expected columns: item_id, item
    id_to_title = dict(zip(mapping_df["item_id"], mapping_df["item"]))
    title_to_id = {title: item_id for item_id, title in id_to_title.items()}
    return id_to_title, title_to_id


def load_ground_truth(csv_path: str) -> dict:
    """
    Load ground truth positive items for each user.
    Expects a CSV with columns: user_id, positive_item, where positive_item is a stringified list.
    Returns:
        dict: {user_id: set(positive_item_ids), ...}
    """
    gt_df = pd.read_csv(csv_path)
    ground_truth = {}
    for _, row in gt_df.iterrows():
        user = row["user_id"]
        pos_items = ast.literal_eval(row["positive_item"])
        ground_truth[user] = set(pos_items)
    return ground_truth


def preprocess_topk_lists(topk_dict: dict, ground_truth: dict, n: int = 5):
    """
    Preprocess the top-k lists to ensure user has more than n positive items.
    """
    filtered_topk_dict = {}
    for user_id, (rec_titles, rel_scores) in topk_dict.items():
        gt_ids = ground_truth.get(user_id, set())
        if len(gt_ids) > n:
            filtered_topk_dict[user_id] = (rec_titles, rel_scores)
    return filtered_topk_dict


# ------------------------------------------
# 3. Metric Computation Utility for Recall, NDCG, and Hit Rate
# ------------------------------------------
def compute_metrics(
    diversified_results: dict, ground_truth: dict, title_to_id: dict, k: int = 10
):
    """
    Compute the average Recall@K, NDCG@K, and Hit Rate@K across all users.

    Parameters:
        diversified_results (dict): {user_id: (list_of_titles, list_of_scores), ...}
        ground_truth (dict): {user_id: set(positive_item_ids), ...}
        title_to_id (dict): Mapping from item title to item id.
        k (int): Top-k cutoff for the metrics.

    Returns:
        tuple: (avg_recall, avg_ndcg, avg_hit_rate)
    """
    all_recalls, all_ndcgs, all_hits, all_precision = [], [], [], []

    for user_id, (rec_titles, _) in diversified_results.items():
        gt_ids = ground_truth.get(user_id, set())
        recall, ndcg, hit, precision = compute_user_metrics(
            rec_titles, gt_ids, title_to_id, k=k
        )
        all_recalls.append(recall)
        all_ndcgs.append(ndcg)
        all_hits.append(hit)
        all_precision.append(precision)

    avg_recall = np.mean(all_recalls, dtype=np.float64)
    avg_ndcg = np.mean(all_ndcgs, dtype=np.float64)
    avg_hit_rate = np.mean(all_hits, dtype=np.float64)
    avg_precision = np.mean(all_precision, dtype=np.float64)
    return avg_recall, avg_ndcg, avg_hit_rate, avg_precision


# ------------------------------------------
# 4. Main Function
# Assuming diversified_results is your dictionary with user recommendations
# Format: { user_id: (list_of_titles, list_of_scores), ... }
# For example:
# diversified_results = {
#     1: (['Scout, The (1994)'], [2.1710939407348633]),
#     2: (['Scout, The (1994)'], [2.1220200061798096]),
#     3: (["Devil's Own, The (1997)", "Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964)"], [1.93, 1.68]),
#     ...
# }
# ------------------------------------------
def main():
    # Define paths
    data_path = "topk_data/ml100k"
    topk_file = f"{data_path}/CMF_top100.pkl"
    mapping_file = f"{data_path}/target_item_id_mapping.csv"
    ground_truth_file = f"{data_path}/uid2positive_item.csv"
    k = 10

    # Load item mapping and ground truth for top-k metrics
    _, title_to_id = load_item_mapping(mapping_file)
    ground_truth = load_ground_truth(ground_truth_file)

    # Load the top-k recommendations (dictionary)
    topk_dict = pickle.load(open(topk_file, "rb"))

    print("Before diversification and preprocess:")

    # Compute Recall, NDCG, and Hit Rate
    avg_recall, avg_ndcg, avg_hit_rate, avg_precision = compute_metrics(
        topk_dict, ground_truth, title_to_id, k=k
    )
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Hit Rate@{k}: {avg_hit_rate:.4f}")

    # Preprocess the top-k lists to ensure user has more than n positive items
    topk_dict = preprocess_topk_lists(topk_dict, ground_truth, n=5)

    # Initialize the diversifier (using BSwapDiversifier as an example)
    diversifier = BSwapDiversifier(
        model_name="all-MiniLM-L6-v2", device="cuda", batch_size=128, theta=0.7
    )

    print("Before diversification:")

    avg_ild = compute_average_ild(topk_dict, diversifier.embedder, topk=10)
    print(f"ILD@{k}: {avg_ild:.4f}")
    # Compute Recall, NDCG, and Hit Rate
    avg_recall, avg_ndcg, avg_hit_rate, avg_precision = compute_metrics(
        topk_dict, ground_truth, title_to_id, k=k
    )
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Hit Rate@{k}: {avg_hit_rate:.4f}")

    # Run diversification
    diversified_results = run_diversification(topk_dict, diversifier, top_k=10)

    # Average num of items in the diversified results
    avg_num_items = np.mean([len(x[0]) for x in diversified_results.values()])
    print(f"Average number of items in diversified results: {avg_num_items:.2f}")

    print("After diversification:")
    avg_ild = compute_average_ild(diversified_results, diversifier.embedder, topk=10)
    print(f"ILD@{k}: {avg_ild:.4f}")

    # Compute Recall, NDCG, and Hit Rate
    avg_recall, avg_ndcg, avg_hit_rate, avg_precision = compute_metrics(
        diversified_results, ground_truth, title_to_id, k=k
    )
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Hit Rate@{k}: {avg_hit_rate:.4f}")


if __name__ == "__main__":
    main()

import pickle
import numpy as np


from algorithms.bswap import BSwapDiversifier
from algorithms.base import BaseDiversifier

from utils import (
    compute_average_ild_batched,
    evaluate_recommendation_metrics,
    load_data_and_convert,
    precompute_title_embeddings,
    create_relevance_lists,
)


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

    # Precompute embeddings for all unique titles
    title2embedding = precompute_title_embeddings(rankings, diversifier.embedder)

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

        # Run diversification using precomputed embeddings.
        diversified_items = diversifier.diversify(
            items, top_k=top_k, title2embedding=title2embedding
        )

        # Extract diversified titles and scores (ignore the dummy index column)
        diversified_titles = diversified_items[:, 1].tolist()
        diversified_scores = [float(x) for x in diversified_items[:, 2]]

        diversified_dict[user_id] = (diversified_titles, diversified_scores)

    return diversified_dict


def main():
    data_path = "topk_data/amazon14"
    item_mappings = f"{data_path}/final.csv"
    test_samples = f"{data_path}/amz_14_final_samples.csv"

    # The converted data is in the form of:
    # [[pos_item, neg_item1, neg_item2, ...], ...] for the user_id=1 the list index is 0.
    converted_data = load_data_and_convert(test_samples, item_mappings)

    # Extract pos items for each user from the converted data
    pos_items = [x[0] for x in converted_data]

    # Load ranked top100 items
    # shape = [num_users, 100]. The value of 0 is the positive item.
    topk_list = pickle.load(open("topk_data/amazon14/amz-topk_iid_list.pkl", "rb"))
    topk_scores = pickle.load(open("topk_data/amazon14/amz-topk_score.pkl", "rb"))

    # For each user, convert indices to item titles
    rankings = {}
    for idx, (items, scores) in enumerate(zip(topk_list, topk_scores)):
        user_id = idx + 1
        titles = [converted_data[idx][item] for item in items]
        rankings[user_id] = (titles, scores.tolist())

    # Get the relevance lists
    relevance_lists = create_relevance_lists(rankings, pos_items)

    mean_prec, mean_rec, mean_hit, mean_ndcg = evaluate_recommendation_metrics(
        relevance_lists, 10
    )

    # Init the diversifier
    diversifier = BSwapDiversifier(
        model_name="all-MiniLM-L6-v2", device="cuda", batch_size=40960, theta=0.7
    )

    # Compute ILD for the top-10 recommendations
    avg_ild = compute_average_ild_batched(rankings, diversifier.embedder, topk=10)

    print("Before diversification:")
    print(f"ILD@10: {avg_ild:.4f}")
    print(f"NDCG@10: {mean_ndcg:.4f}")
    print(f"Hit@10: {mean_hit:.4f}")
    print(f"Recall@10: {mean_rec:.4f}")
    print(f"Precision@10: {mean_prec:.4f}")

    # Run diversification
    diversified_results = run_diversification(rankings, diversifier, top_k=10)

    # Get the relevance lists
    relevance_lists = create_relevance_lists(diversified_results, pos_items)

    mean_prec, mean_rec, mean_hit, mean_ndcg = evaluate_recommendation_metrics(
        relevance_lists, 10
    )

    # Compute ILD for the top-10 recommendations
    avg_ild = compute_average_ild_batched(
        diversified_results, diversifier.embedder, topk=10
    )

    print("After diversification:")
    print(f"ILD@10: {avg_ild:.4f}")
    print(f"NDCG@10: {mean_ndcg:.4f}")
    print(f"Hit@10: {mean_hit:.4f}")
    print(f"Recall@10: {mean_rec:.4f}")
    print(f"Precision@10: {mean_prec:.4f}")


if __name__ == "__main__":
    main()

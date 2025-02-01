import pickle

from algorithms.sy import SYDiversifier
from algorithms.motley import MotleyDiversifier
from algorithms.mmr import MMRDiversifier
from algorithms.swap import SwapDiversifier
from algorithms.sy import SYDiversifier
from algorithms.msd import MaxSumDiversifier
from utils import compute_average_ild
import numpy as np
from algorithms.base import BaseDiversifier

def run_diversification(
    rankings: dict, diversifier: BaseDiversifier, top_k: int = 10
) -> dict:
    """
    Run the diversification algorithm on recommendation titles and return
    the diversified output as a dictionary with the same structure as the input.

    The diversification algorithm expects an np.ndarray with at least three columns:
      - Column 0: a dummy index representing the original ranking position.
      - Column 1: the title (string).
      - Column 2: the relevance score (float).

    Parameters:
        rankings (dict): Rankings in the form of a dict
                         {user_id: (titles: [str], relevance_scores: [float]), ...}.
        diversifier (BaseDiversifier): An instance of a diversification algorithm (e.g., MMRDiversifier).
        top_k (int): Number of top items to select after diversification.

    Returns:
        dict: A dictionary where each key is a user_id and the value is a tuple:
              (diversified_titles: [str], diversified_scores: [float]).
              The structure matches the undiversified version.
    """
    diversified_dict = {}

    for user_id, (titles, relevance_scores) in rankings.items():
        if len(titles) != len(relevance_scores):
            raise ValueError(f"User {user_id}: Number of titles and relevance scores do not match.")

        # Create an array with rows: [dummy_index, title, relevance_score]
        items = np.array(
            [[i, title, float(relevance_scores[i])] for i, title in enumerate(titles)],
            dtype=object,
        )

        # Run the diversification algorithm for this user.
        diversified_items = diversifier.diversify(items,
                                                  top_k=top_k
                                                  )

        # Extract diversified titles and scores (ignore the dummy index column)
        diversified_titles = diversified_items[:, 1].tolist()
        diversified_scores = [float(x) for x in diversified_items[:, 2]]

        diversified_dict[user_id] = (diversified_titles, diversified_scores)

    return diversified_dict


# ----------------------------
# Example format for running the diversification algorithm.:
#
# top-k dictionary:
# topk_dict = {
#    "user1": (["title1", "title2", "title3", "title4", "title5"], [0.9, 0.8, 0.7, 0.6, 0.5]),
#    "user2": (["title1", "title2", "title3", "title4", "title5"], [0.9, 0.8, 0.7, 0.6, 0.5]),
#    ...
# }
# ----------------------------

if __name__ == "__main__":
    # Example top-k dictionary for many users.
    topk_dict = pickle.load(open("topk_data/ml100k/CMF_topk.pkl", "rb"))

    diversifier = SYDiversifier(
        model_name="all-MiniLM-L6-v2", device="cuda", batch_size=128, threshold=0.1
    )

    # print("Before Diversification:")
    # # Compute and print the average ILD over all users.
    # avg_ild = compute_average_ild(topk_dict, diversifier.embedder, topk=3)
    # print(f"Average ILD: {avg_ild:.4f}")


    diversified_results = run_diversification(topk_dict, diversifier, top_k=3)

    # Avg num items in top-k
    avg_num_items = np.mean([len(x[0]) for x in diversified_results.values()])
    print("After Diversification:")
    print(f"Avg num items in top-k: {avg_num_items:.4f}")
    # Compute and print the average ILD over all users.
    avg_ild = compute_average_ild(diversified_results, diversifier.embedder, topk=3)
    print(f"Average ILD: {avg_ild:.4f}")

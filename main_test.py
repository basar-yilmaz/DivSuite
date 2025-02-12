import pickle
import numpy as np
import csv
import logging
import matplotlib.pyplot as plt

from algorithms.bswap import BSwapDiversifier
from utils import (
    compute_average_ild_batched,  # Updated to accept an optional 'precomputed_embeddings' argument.
    evaluate_recommendation_metrics,
    load_data_and_convert,
    precompute_title_embeddings,
    create_relevance_lists,
)
from embedders.ste_embedder import (
    STEmbedder,
)  # Ensure this is your STEmbedder implementation.

# Set up logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def advanced_metric_tracking_pipeline(
    diversifier_cls,
    diversifier_param_name: str,
    param_start: float,
    param_end: float,
    param_step: float,
    threshold_drop: float = 0.1,
    top_k: int = 10,
):
    """
    Vary a generic diversification algorithm parameter and track metrics,
    stopping if NDCG drops more than the threshold relative to the non-diversified baseline.

    Parameters:
      diversifier_cls: The diversification algorithm class (e.g. BSwapDiversifier).
      diversifier_param_name (str): The name of the parameter to vary (e.g., "theta" or "lambda").
      param_start (float): Starting value of the parameter.
      param_end (float): Ending value of the parameter.
      param_step (float): Step change for the parameter.
      threshold_drop (float): Fractional decrease in NDCG (from baseline) at which to stop (e.g., 0.1 for 10%).
      top_k (int): Number of top recommendations to evaluate.
    """
    logging.info("=== Starting Metric Tracking Pipeline ===")

    # ---------------------------
    # Load Data and Build Rankings
    # ---------------------------
    data_path = "topk_data/amazon14"
    item_mappings = f"{data_path}/final.csv"
    test_samples = f"{data_path}/amz_14_final_samples.csv"

    # Each row: [pos_item, neg_item1, neg_item2, ...]
    converted_data = load_data_and_convert(test_samples, item_mappings)
    pos_items = [row[0] for row in converted_data]

    # Load ranked top-100 items and scores.
    topk_list = pickle.load(open(f"{data_path}/amz-topk_iid_list.pkl", "rb"))
    topk_scores = pickle.load(open(f"{data_path}/amz-topk_score.pkl", "rb"))

    # Build rankings: {user_id: (titles: [str], relevance_scores: [float]), ...}
    rankings = {}
    for idx, (items, scores) in enumerate(zip(topk_list, topk_scores)):
        user_id = idx + 1
        titles = [converted_data[idx][item] for item in items]
        rankings[user_id] = (titles, scores.tolist())

    # ---------------------------
    # Compute Baseline (Non-Diversified) Metrics
    # ---------------------------
    # These are computed directly on the original rankings.
    baseline_relevance = create_relevance_lists(rankings, pos_items)
    baseline_prec, baseline_rec, baseline_hit, baseline_ndcg = (
        evaluate_recommendation_metrics(baseline_relevance, top_k)
    )

    # Precompute embeddings once using a baseline embedder instance.
    embedder = STEmbedder(
        model_name="all-MiniLM-L6-v2", device="cuda", batch_size=40960
    )
    precomputed_embeddings = precompute_title_embeddings(rankings, embedder)

    # Compute ILD for the original ranking using precomputed embeddings.
    baseline_ild = compute_average_ild_batched(
        rankings,
        embedder,
        topk=top_k,
        precomputed_embeddings=precomputed_embeddings,
    )
    logging.info(
        "Baseline (Non-diversified): NDCG@%d=%.4f, ILD@%d=%.4f",
        top_k,
        baseline_ndcg,
        top_k,
        baseline_ild,
    )

    # ---------------------------
    # Diversification Loop (Varying the Generic Parameter)
    # ---------------------------
    results = []  # To store metrics for each parameter value.
    param_values = np.arange(param_start, param_end + param_step / 2, param_step)
    for param_value in param_values:
        logging.info(
            "Running diversification with %s=%.2f", diversifier_param_name, param_value
        )

        # Create diversifier instance with the current parameter value.
        diversifier_kwargs = {diversifier_param_name: param_value}
        diversifier = diversifier_cls(
            embedder=embedder,
            **diversifier_kwargs,
        )

        # Run diversification using the precomputed embeddings.
        diversified_results = run_diversification(
            rankings,
            diversifier,
            top_k=top_k,
            precomputed_embeddings=precomputed_embeddings,
        )

        # Compute evaluation metrics on the diversified results.
        relevance_lists_div = create_relevance_lists(diversified_results, pos_items)
        prec, rec, hit, ndcg = evaluate_recommendation_metrics(
            relevance_lists_div, top_k
        )
        ild = compute_average_ild_batched(
            diversified_results,
            diversifier.embedder,
            topk=top_k,
            precomputed_embeddings=precomputed_embeddings,
        )

        logging.info(
            "%s=%.2f: NDCG@%d=%.4f, ILD@%d=%.4f, Hit@%d=%.4f, Recall@%d=%.4f, Precision@%d=%.4f",
            diversifier_param_name,
            param_value,
            top_k,
            ndcg,
            top_k,
            ild,
            top_k,
            hit,
            top_k,
            rec,
            top_k,
            prec,
        )

        results.append(
            {
                diversifier_param_name: param_value,
                "ndcg": ndcg,
                "ild": ild,
                "hit": hit,
                "recall": rec,
                "precision": prec,
            }
        )

        # Stop early if NDCG has dropped more than the threshold.
        if ndcg < baseline_ndcg * (1 - threshold_drop):
            logging.info(
                "Stopping early: NDCG dropped more than %.0f%% from baseline.",
                threshold_drop * 100,
            )
            break

    # ---------------------------
    # Log Metrics to CSV
    # ---------------------------
    csv_filename = "diversification_metrics.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            diversifier_param_name,
            "ndcg",
            "ild",
            "hit",
            "recall",
            "precision",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    logging.info("Metrics logged to CSV file: %s", csv_filename)

    # ---------------------------
    # Plot the Metrics Using Matplotlib
    # ---------------------------
    param_vals = [res[diversifier_param_name] for res in results]
    ndcg_vals = [res["ndcg"] for res in results]
    ild_vals = [res["ild"] for res in results]

    plt.figure(figsize=(10, 6))
    plt.plot(param_vals, ndcg_vals, marker="o", label=f"NDCG@{top_k}")
    plt.plot(param_vals, ild_vals, marker="x", label=f"ILD@{top_k}")
    plt.xlabel(f"{diversifier_param_name.capitalize()} Parameter")
    plt.ylabel("Metric Value")
    plt.title(
        f"Recommendation Metrics vs. {diversifier_param_name.capitalize()} Parameter"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = "diversification_metrics.png"
    plt.savefig(plot_filename)
    # plt.show()
    logging.info("Plot saved to %s", plot_filename)

    logging.info("=== Metric Tracking Pipeline Completed ===")


def run_diversification(
    rankings: dict, diversifier, top_k: int = 10, precomputed_embeddings: dict = None
) -> dict:
    """
    Diversify recommendation lists using the provided diversifier.
    If 'precomputed_embeddings' is provided, it will be reused instead of recalculating embeddings.
    """
    diversified_dict = {}
    title2embedding = (
        precomputed_embeddings
        if precomputed_embeddings is not None
        else precompute_title_embeddings(rankings, diversifier.embedder)
    )

    for user_id, (titles, relevance_scores) in rankings.items():
        if len(titles) != len(relevance_scores):
            raise ValueError(
                f"User {user_id}: Number of titles and relevance scores do not match."
            )
        # Create an array of items: [dummy_index, title, relevance_score]
        items = np.array(
            [[i, title, float(relevance_scores[i])] for i, title in enumerate(titles)],
            dtype=object,
        )
        diversified_items = diversifier.diversify(
            items, top_k=top_k, title2embedding=title2embedding
        )
        diversified_titles = diversified_items[:, 1].tolist()
        diversified_scores = [float(x) for x in diversified_items[:, 2]]
        diversified_dict[user_id] = (diversified_titles, diversified_scores)

    return diversified_dict


if __name__ == "__main__":
    # Example usage with BSwapDiversifier (varying the "theta" parameter).
    advanced_metric_tracking_pipeline(
        diversifier_cls=BSwapDiversifier,
        diversifier_param_name="theta",
        param_start=1.0,
        param_end=0.4,
        param_step=-0.05,
        threshold_drop=0.1,
        top_k=10,
    )

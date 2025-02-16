import pickle
import numpy as np
import csv
import logging
import matplotlib.pyplot as plt
import os
import datetime

from algorithms.motley import MotleyDiversifier

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

    # Create a folder to store results (CSV and plots)
    # Determine the folder name based on the diversifier class name.
    diversifier_name = diversifier_cls.__name__.replace("Diversifier", "").lower()
    results_folder = f"results_{diversifier_name}"
    os.makedirs(results_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
    baseline_relevance = create_relevance_lists(rankings, pos_items)
    baseline_prec, baseline_rec, baseline_hit, baseline_ndcg = (
        evaluate_recommendation_metrics(baseline_relevance, top_k)
    )

    # Precompute embeddings once using a baseline embedder instance.
    embedder = STEmbedder(
        model_name="all-MiniLM-L6-v2", device="cuda", batch_size=40960
    )
    precomputed_embeddings = precompute_title_embeddings(rankings, embedder)

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
                "ndcg": round(ndcg, 4),
                "ild": round(ild, 4),
                "hit": round(hit, 4),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
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
    csv_filename = os.path.join(
        results_folder, f"diversification_metrics_{timestamp}.csv"
    )
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
        # Write with 4 decimal points formatting.
        for res in results:
            writer.writerow(
                {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in res.items()}
            )
    logging.info("Metrics logged to CSV file: %s", csv_filename)

    # ---------------------------
    # Create Plots
    # ---------------------------
    param_vals = [res[diversifier_param_name] for res in results]
    ndcg_vals = [res["ndcg"] for res in results]
    ild_vals = [res["ild"] for res in results]

    # Plot 1: Subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(param_vals, ndcg_vals, marker="o", color="blue")
    axes[0].set_xlabel(f"{diversifier_param_name.capitalize()} Parameter")
    axes[0].set_ylabel(f"NDCG@{top_k}")
    axes[0].set_title("NDCG vs. " + diversifier_param_name.capitalize())
    axes[0].grid(True)

    axes[1].plot(param_vals, ild_vals, marker="x", color="red")
    axes[1].set_xlabel(f"{diversifier_param_name.capitalize()} Parameter")
    axes[1].set_ylabel(f"ILD@{top_k}")
    axes[1].set_title("ILD vs. " + diversifier_param_name.capitalize())
    axes[1].grid(True)

    plt.tight_layout()
    subplot_filename = os.path.join(
        results_folder, f"diversification_subplots_{timestamp}.png"
    )
    plt.savefig(subplot_filename)
    plt.close(fig)
    logging.info("Subplot saved to %s", subplot_filename)

    # Plot 2: Combined plot (both metrics in one graph)
    plt.figure(figsize=(10, 6))
    plt.plot(param_vals, ndcg_vals, marker="o", label=f"NDCG@{top_k}", color="blue")
    plt.plot(param_vals, ild_vals, marker="x", label=f"ILD@{top_k}", color="red")
    plt.xlabel(f"{diversifier_param_name.capitalize()} Parameter")
    plt.ylabel("Metric Value")
    plt.title(
        f"Recommendation Metrics vs. {diversifier_param_name.capitalize()} Parameter"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    combined_plot_filename = os.path.join(
        results_folder, f"diversification_metrics_{timestamp}.png"
    )
    plt.savefig(combined_plot_filename)
    logging.info("Combined plot saved to %s", combined_plot_filename)

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
        diversifier_cls=MotleyDiversifier,
        diversifier_param_name="theta_",
        param_start=0,
        param_end=1.0,
        param_step=0.05,
        threshold_drop=0.1,
        top_k=10,
    )

import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path

from algorithms.mmr import MMRDiversifier
from algorithms.motley import MotleyDiversifier
from algorithms.bswap import BSwapDiversifier
from algorithms.clt import CLTDiversifier
from algorithms.msd import MaxSumDiversifier
from algorithms.swap import SwapDiversifier
from algorithms.sy import SYDiversifier

from config_parser import get_config
from utils import (
    compute_average_ild_batched,
    compute_average_category_ild_batched,
    evaluate_recommendation_metrics,
    load_data_and_convert,
    load_movie_categories,
    precompute_title_embeddings,
    create_relevance_lists,
)
from embedders.ste_embedder import STEmbedder
from logger import get_logger

# Set up logging configuration
logger = get_logger(__name__)


def advanced_metric_tracking_pipeline(config):
    """
    Vary a generic diversification algorithm parameter and track metrics,
    stopping if NDCG drops more than the threshold relative to the non-diversified baseline.
    """
    logger.info("=== Starting Metric Tracking Pipeline ===")

    # Get experiment parameters from config
    diversifier_cls = globals()[config["experiment"]["diversifier"]]
    diversifier_param_name = config["experiment"]["param_name"]
    param_start = config["experiment"]["param_start"]
    param_end = config["experiment"]["param_end"]
    param_step = config["experiment"]["param_step"]
    threshold_drop = config["experiment"]["threshold_drop"]
    top_k = config["experiment"]["top_k"]
    use_category_ild = config["experiment"].get("use_category_ild", False)

    # Create results directory
    diversifier_name = diversifier_cls.__name__.replace("Diversifier", "").lower()
    results_folder = f"results_{diversifier_name}"
    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------
    # Load Data and Build Rankings
    # ---------------------------
    data_path = Path(config["data"]["base_path"])
    item_mappings = data_path / config["data"]["item_mappings"]
    test_samples = data_path / config["data"]["test_samples"]
    topk_list_path = data_path / config["data"]["topk_list"]
    topk_scores_path = data_path / config["data"]["topk_scores"]

    # Only load categories if needed
    categories_data = None
    if use_category_ild:
        categories_path = data_path / config["data"]["movie_categories"]
        categories_data = load_movie_categories(str(categories_path))

    # Load and convert data
    converted_data = load_data_and_convert(str(test_samples), str(item_mappings))
    pos_items = [row[0] for row in converted_data]

    # Load ranked items and scores
    topk_list = pickle.load(open(topk_list_path, "rb"))
    topk_scores = pickle.load(open(topk_scores_path, "rb"))

    # Build rankings
    rankings = {}
    for idx, (items, scores) in enumerate(zip(topk_list, topk_scores)):
        user_id = idx + 1
        titles = [converted_data[idx][item] for item in items]
        rankings[user_id] = (titles, scores.tolist())

    # ---------------------------
    # Initialize Embedder
    # ---------------------------
    embedder = STEmbedder(
        model_name=config["embedder"]["model_name"],
        device=config["embedder"]["device"],
        batch_size=config["embedder"]["batch_size"],
    )

    # ---------------------------
    # Compute Baseline (Non-Diversified) Metrics
    # ---------------------------
    baseline_relevance = create_relevance_lists(rankings, pos_items)
    baseline_prec, baseline_rec, baseline_hit, baseline_ndcg = (
        evaluate_recommendation_metrics(baseline_relevance, top_k)
    )

    # Precompute embeddings once using a baseline embedder instance.
    precomputed_embeddings = precompute_title_embeddings(rankings, embedder)

    # Calculate embedding-based ILD
    baseline_emb_ild = compute_average_ild_batched(
        rankings,
        embedder,
        topk=top_k,
        precomputed_embeddings=precomputed_embeddings,
    )

    # Calculate category-based ILD if enabled
    baseline_cat_ild = None
    if use_category_ild:
        baseline_cat_ild = compute_average_category_ild_batched(
            rankings,
            categories_data,
            topk=top_k,
        )

    # Log baseline metrics
    if use_category_ild:
        logger.info(
            "Baseline (Non-diversified): NDCG@%d=%.4f, Emb-ILD@%d=%.4f, Cat-ILD@%d=%.4f",
            top_k,
            baseline_ndcg,
            top_k,
            baseline_emb_ild,
            top_k,
            baseline_cat_ild,
        )
    else:
        logger.info(
            "Baseline (Non-diversified): NDCG@%d=%.4f, Emb-ILD@%d=%.4f",
            top_k,
            baseline_ndcg,
            top_k,
            baseline_emb_ild,
        )

    # ---------------------------
    # Diversification Loop (Varying the Generic Parameter)
    # ---------------------------
    results = []  # To store metrics for each parameter value.
    param_values = np.arange(param_start, param_end + param_step / 2, param_step)
    for param_value in param_values:
        logger.info(
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

        # Calculate embedding-based ILD
        emb_ild = compute_average_ild_batched(
            diversified_results,
            diversifier.embedder,
            topk=top_k,
            precomputed_embeddings=precomputed_embeddings,
        )

        # Calculate category-based ILD if enabled
        cat_ild = None
        if use_category_ild:
            cat_ild = compute_average_category_ild_batched(
                diversified_results,
                categories_data,
                topk=top_k,
            )

        ndcg_decrease_pct = (baseline_ndcg - ndcg) / baseline_ndcg

        # Log metrics based on whether category ILD is enabled
        if use_category_ild:
            logger.info(
                "%s=%.2f: NDCG@%d=%.4f (%.2f%% decrease), Emb-ILD@%d=%.4f, Cat-ILD@%d=%.4f",
                diversifier_param_name,
                param_value,
                top_k,
                ndcg,
                ndcg_decrease_pct * 100,
                top_k,
                emb_ild,
                top_k,
                cat_ild,
            )
        else:
            logger.info(
                "%s=%.2f: NDCG@%d=%.4f (%.2f%% decrease), Emb-ILD@%d=%.4f",
                diversifier_param_name,
                param_value,
                top_k,
                ndcg,
                ndcg_decrease_pct * 100,
                top_k,
                emb_ild,
            )

        # Store results
        result_dict = {
            diversifier_param_name: param_value,
            "ndcg": round(ndcg, 4),
            "ndcg_drop": round(ndcg_decrease_pct * 100, 4),
            "emb_ild": round(emb_ild, 4),
            "hit": round(hit, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
        }
        if use_category_ild:
            result_dict["cat_ild"] = round(cat_ild, 4)

        results.append(result_dict)

        # Stop when NDCG drops by threshold_drop
        if ndcg_decrease_pct > threshold_drop:
            if use_category_ild:
                logger.warning(
                    "Stopping: NDCG dropped by %.2f%%. Best Emb-ILD before drop: %.4f, Best Cat-ILD before drop: %.4f",
                    ndcg_decrease_pct * 100,
                    results[-2]["emb_ild"] if len(results) > 1 else baseline_emb_ild,
                    results[-2]["cat_ild"] if len(results) > 1 else baseline_cat_ild,
                )
            else:
                logger.warning(
                    "Stopping: NDCG dropped by %.2f%%. Best Emb-ILD before drop: %.4f",
                    ndcg_decrease_pct * 100,
                    results[-2]["emb_ild"] if len(results) > 1 else baseline_emb_ild,
                )
            break

    # ---------------------------
    # Log Metrics to CSV
    # ---------------------------
    csv_filename = os.path.join(
        results_folder, f"diversification_metrics_{timestamp}.csv"
    )

    # Define fieldnames based on whether category ILD is enabled
    fieldnames = [
        diversifier_param_name,
        "ndcg",
        "ndcg_drop",
        "emb_ild",
        "hit",
        "recall",
        "precision",
    ]
    if use_category_ild:
        fieldnames.insert(4, "cat_ild")

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for res in results:
            writer.writerow(
                {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in res.items()}
            )
    logger.info("Metrics logged to CSV file: %s", csv_filename)

    # ---------------------------
    # Create Plots
    # ---------------------------
    param_vals = [res[diversifier_param_name] for res in results]
    ndcg_vals = [res["ndcg"] for res in results]
    emb_ild_vals = [res["emb_ild"] for res in results]

    # Create figure with appropriate number of y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot NDCG on primary y-axis
    color1 = "#1f77b4"  # Blue
    ax1.set_xlabel(f"{diversifier_param_name.capitalize()} Parameter")
    ax1.set_ylabel("NDCG", color=color1)
    line1 = ax1.plot(param_vals, ndcg_vals, color=color1, marker="o", label="NDCG")
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create secondary y-axis and plot Embedding ILD
    ax2 = ax1.twinx()
    color2 = "#ff7f0e"  # Orange
    ax2.set_ylabel("Embedding ILD", color=color2)
    line2 = ax2.plot(
        param_vals, emb_ild_vals, color=color2, marker="x", label="Emb-ILD"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]

    # Add Category ILD if enabled
    if use_category_ild:
        cat_ild_vals = [res["cat_ild"] for res in results]
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color3 = "#2ca02c"  # Green
        ax3.set_ylabel("Category ILD", color=color3)
        line3 = ax3.plot(
            param_vals, cat_ild_vals, color=color3, marker="s", label="Cat-ILD"
        )
        ax3.tick_params(axis="y", labelcolor=color3)
        lines += line3
        labels.append("Cat-ILD")

    # Add legend
    ax1.legend(lines, labels, loc="center right")

    # Add grid
    ax1.grid(True, alpha=0.3)

    plt.title(f"{diversifier_cls.__name__} Performance")
    plt.tight_layout()

    # Save plot
    plot_filename = os.path.join(
        results_folder, f"diversification_metrics_{timestamp}.png"
    )
    plt.savefig(plot_filename)
    logger.info("Plot saved to %s", plot_filename)
    plt.close()

    logger.info("=== Metric Tracking Pipeline Completed ===")


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
    config = get_config()
    advanced_metric_tracking_pipeline(config)

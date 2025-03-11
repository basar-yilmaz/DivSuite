"""Module for visualization and results logging."""

import os
import csv
import matplotlib.pyplot as plt
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def log_metrics_to_csv(
    results: List[dict],
    results_folder: str,
    timestamp: str,
    experiment_params: dict,
) -> str:
    """
    Log metrics to CSV file.

    Args:
        results: List of result dictionaries.
        results_folder: Directory to save results.
        timestamp: Timestamp string for filename.
        experiment_params: Experiment parameters.

    Returns:
        str: Path to the created CSV file.
    """
    threshold = experiment_params.get("threshold_drop", "NA")
    csv_filename = os.path.join(
        results_folder, f"diversification_metrics_drop{threshold}_{timestamp}.csv"
    )

    fieldnames = [
        experiment_params["diversifier_param_name"],
        "ndcg",
        "ndcg_drop",
        "mrr",
        "emb_ild",
        "hit",
        "recall",
        "precision",
    ]
    if experiment_params["use_category_ild"]:
        fieldnames.insert(5, "cat_ild")

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for res in results:
            writer.writerow(
                {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in res.items()}
            )
    logger.info("Metrics logged to CSV file: %s", csv_filename)

    return csv_filename


def create_plots(
    results: List[dict],
    results_folder: str,
    timestamp: str,
    experiment_params: dict,
    baseline_metrics: dict,
) -> str:
    """
    Create and save visualization plots.

    Args:
        results: List of result dictionaries.
        results_folder: Directory to save plots.
        timestamp: Timestamp string for filename.
        experiment_params: Experiment parameters.
        baseline_metrics: Baseline metrics for comparison.

    Returns:
        str: Path to the created plot file.
    """
    if not results:
        logger.warning("No results to plot")
        return ""

    param_vals = [res[experiment_params["diversifier_param_name"]] for res in results]
    ndcg_vals = [res["ndcg"] for res in results]
    emb_ild_vals = [res["emb_ild"] for res in results]

    # Calculate marker frequency
    def get_markevery(data_length):
        return max(1, data_length // 10) if data_length > 0 else 1

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot NDCG
    color1 = (31 / 255, 119 / 255, 180 / 255)  # Blue
    ax1.set_xlabel(
        f"{experiment_params['diversifier_param_name'].capitalize()} Parameter"
    )
    ax1.set_ylabel("NDCG", color=color1)
    line1 = ax1.plot(
        param_vals,
        ndcg_vals,
        color=color1,
        marker="o",
        markevery=get_markevery(len(param_vals)),
        label="NDCG",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Plot Embedding ILD
    ax2 = ax1.twinx()
    color2 = (255 / 255, 127 / 255, 14 / 255)  # Orange
    ax2.set_ylabel("Embedding ILD", color=color2)
    line2 = ax2.plot(
        param_vals,
        emb_ild_vals,
        color=color2,
        marker="x",
        markevery=get_markevery(len(param_vals)),
        label="Emb-ILD",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]

    # Add Category ILD if enabled
    if experiment_params["use_category_ild"]:
        cat_ild_vals = [res["cat_ild"] for res in results]
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color3 = (44 / 255, 160 / 255, 44 / 255)  # Green
        ax3.set_ylabel("Category ILD", color=color3)
        line3 = ax3.plot(
            param_vals,
            cat_ild_vals,
            color=color3,
            marker="s",
            markevery=get_markevery(len(param_vals)),
            label="Cat-ILD",
        )
        lines += line3
        labels.append("Cat-ILD")

    ax1.legend(lines, labels, loc="center left")
    ax1.grid(True, alpha=0.3)

    plt.title(f"{experiment_params['diversifier_cls'].__name__} Performance")
    plt.tight_layout()

    threshold = experiment_params.get("threshold_drop", "NA")
    plot_filename = os.path.join(
        results_folder, f"diversification_metrics_drop{threshold}_{timestamp}.png"
    )
    plt.savefig(plot_filename)
    logger.info("Plot saved to %s", plot_filename)
    plt.close()

    return plot_filename

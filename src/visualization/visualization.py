"""Module for visualization and results logging."""

import csv
import os

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from src.utils.logger import get_logger

plt.style.use(["science", "ieee", "no-latex"])
logger = get_logger(__name__)


def log_metrics_to_csv(
    results: list[dict],
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
    results: list[dict],
    results_folder: str,
    timestamp: str,
    experiment_params: dict,
    baseline_metrics: dict,
) -> str:
    if not results:
        logger.warning("No results to plot")
        return ""

    param_vals = [res[experiment_params["diversifier_param_name"]] for res in results]
    ndcg_vals = [res["ndcg"] for res in results]
    emb_ild_vals = [res["emb_ild"] for res in results]

    def get_markevery(data_length):
        return max(1, data_length // 10) if data_length > 0 else 1

    # Colorblind-friendly palette
    colors = {
        "ndcg": "#1f77b4",
        "emb_ild": "#ff7f0e",
        "cat_ild": "#2ca02c",
    }

    fig, ax1 = plt.subplots(figsize=(6.5, 3))
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.15)

    ax1.set_xlabel(r"$\lambda$", fontsize=10)
    ax1.set_ylabel("NDCG", fontsize=10, color=colors["ndcg"])
    line1 = ax1.plot(
        param_vals,
        ndcg_vals,
        color=colors["ndcg"],
        marker="o",
        markersize=3,
        linewidth=1,
        # markevery=get_markevery(len(param_vals)),
        markevery=1,
        label="NDCG",
    )
    ax1.tick_params(axis="y", labelcolor=colors["ndcg"], labelsize=8)
    ax1.tick_params(axis="x", labelsize=8)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Embedding ILD", fontsize=10, color=colors["emb_ild"])
    line2 = ax2.plot(
        param_vals,
        emb_ild_vals,
        color=colors["emb_ild"],
        marker="x",
        markersize=4,
        linewidth=1,
        # markevery=get_markevery(len(param_vals)),
        markevery=1,
        label="Emb-ILD",
    )
    ax2.tick_params(axis="y", labelcolor=colors["emb_ild"], labelsize=8)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]

    if experiment_params.get("use_category_ild", False):
        cat_ild_vals = [res["cat_ild"] for res in results]
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.set_ylabel("Category ILD", fontsize=10, color=colors["cat_ild"])
        line3 = ax3.plot(
            param_vals,
            cat_ild_vals,
            color=colors["cat_ild"],
            marker="s",
            markersize=4,
            linewidth=1,
            # markevery=get_markevery(len(param_vals)),
            markevery=1,
            label="Cat-ILD",
        )
        ax3.tick_params(axis="y", labelcolor=colors["cat_ild"], labelsize=8)
        lines += line3
        labels.append("Cat-ILD")

    ax1.legend(
        lines,
        labels,
        fontsize=8,
        frameon=False,
        loc="upper left",  # vertical, inside top-left
        bbox_to_anchor=(0.02, 0.8),  # fine-tuned near y-axis
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.0,
    )

    axes = [ax1, ax2]
    if experiment_params.get("use_category_ild", False):
        axes.append(ax3)
    for spine in ["top", "right"]:
        for ax in axes:
            ax.spines[spine].set_visible(False)

    algo_name = experiment_params["diversifier_cls"].__name__.replace("Diversifier", "")
    plt.title(f"{algo_name} Algorithm Performance", fontsize=11)
    plt.tight_layout(pad=0.5)

    threshold = experiment_params.get("threshold_drop", "NA")

    # Save as PNG
    png_filename = os.path.join(
        results_folder, f"diversification_metrics_drop{threshold}_{timestamp}.png"
    )
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")

    # Save as vectorized PDF
    pdf_filename = os.path.join(
        results_folder, f"diversification_metrics_drop{threshold}_{timestamp}.pdf"
    )
    # plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    plt.savefig(pdf_filename, dpi=300, bbox_inches="tight")

    logger.info("Plots saved to %s and %s", png_filename, pdf_filename)
    plt.close()

    return pdf_filename

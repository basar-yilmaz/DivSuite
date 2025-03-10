"""Module for handling diversification experiment pipeline."""

from src.data.data_loader import load_experiment_data
from src.utils.experiment_utils import (
    get_experiment_params,
    setup_results_directory,
    initialize_embedder,
)
from src.metrics.metrics_handler import (
    compute_baseline_metrics,
    run_diversification_loop,
)
from src.visualization.visualization import (
    log_metrics_to_csv,
    create_plots,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_diversification_pipeline(config: dict) -> None:
    """
    Run the complete diversification experiment pipeline.

    This function orchestrates the entire experiment pipeline:
    1. Sets up experiment parameters and directories
    2. Loads and processes data
    3. Runs the diversification experiments
    4. Logs results and creates visualizations

    Args:
        config (dict): Configuration dictionary containing all experiment settings
                      including data paths, model parameters, and evaluation metrics.
    """
    logger.info("=== Starting Diversification Pipeline ===")

    # Initialize experiment
    experiment_params = get_experiment_params(config)
    results_folder, timestamp = setup_results_directory(
        experiment_params["diversifier_cls"]
    )

    # Load data
    rankings, pos_items, categories_data = load_experiment_data(
        config, experiment_params["use_category_ild"]
    )

    # Initialize embedder
    embedder = initialize_embedder(config)

    # Run experiment
    baseline_metrics = compute_baseline_metrics(
        rankings=rankings,
        pos_items=pos_items,
        embedder=embedder,
        top_k=experiment_params["top_k"],
        use_category_ild=experiment_params["use_category_ild"],
        categories_data=categories_data,
    )

    results = run_diversification_loop(
        rankings=rankings,
        pos_items=pos_items,
        embedder=embedder,
        experiment_params=experiment_params,
        baseline_metrics=baseline_metrics,
        categories_data=categories_data,
    )

    # Log and visualize results
    csv_filename = log_metrics_to_csv(
        results=results,
        results_folder=results_folder,
        timestamp=timestamp,
        experiment_params=experiment_params,
    )

    create_plots(
        results=results,
        results_folder=results_folder,
        timestamp=timestamp,
        experiment_params=experiment_params,
        baseline_metrics=baseline_metrics,
    )

    logger.info("=== Diversification Pipeline Completed ===")
    return {
        "results": results,
        "baseline_metrics": baseline_metrics,
        "csv_path": csv_filename,
        "results_folder": results_folder,
        "timestamp": timestamp,
    }

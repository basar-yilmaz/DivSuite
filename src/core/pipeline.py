"""Module for handling diversification experiment pipeline."""

import time
import gc  # Added for garbage collection

from src.data.data_loader import load_experiment_data
from src.metrics.metrics_handler import (
    compute_baseline_metrics,
    run_diversification_loop,
)
from src.utils.experiment_utils import (
    get_embedding_params,
    get_experiment_params,
    get_similarity_params,
    initialize_embedder,
    setup_results_directory,
)
from src.utils.logger import get_logger
from src.visualization.visualization import (
    create_plots,
    log_metrics_to_csv,
)

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
    similarity_params = get_similarity_params(config)
    embedding_params = get_embedding_params(config)
    results_folder, timestamp = setup_results_directory(
        experiment_params["diversifier_cls"]
    )

    # Load data
    rankings, pos_items, categories_data, item_id_mapping = load_experiment_data(
        config, experiment_params["use_category_ild"]
    )

    # Initialize embedder
    embedder = initialize_embedder(config, embedding_params)

    # Run experiment
    baseline_metrics = compute_baseline_metrics(
        rankings=rankings,
        pos_items=pos_items,
        embedder=embedder,
        top_k=experiment_params["top_k"],
        use_category_ild=experiment_params["use_category_ild"],
        categories_data=categories_data,
        use_similarity_scores=similarity_params["use_similarity_scores"],
        similarity_scores_path=similarity_params["similarity_scores_path"],
        embedding_params=embedding_params,
        item_id_mapping=item_id_mapping,
    )

    if (
        not embedding_params.get("use_precomputed_embeddings", False)
        and embedder is not None
        and hasattr(embedder, "model")
        and embedder.model is not None
    ):
        logger.info("Releasing embedder model resources to free up memory...")
        try:
            # For STEmbedder, stop multiprocessing pool if it exists and is managed by the embedder instance
            if hasattr(embedder, "pool") and embedder.pool is not None:
                if hasattr(embedder.model, "stop_multi_process_pool"):
                    embedder.model.stop_multi_process_pool(embedder.pool)
                embedder.pool = None

            del embedder.model
            embedder.model = None

            logger.info("Embedder model resources released.")
        except Exception as e:
            logger.warning(f"Could not fully release embedder resources: {e}")
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Called torch.cuda.empty_cache().")
            except ImportError:
                logger.debug(
                    "torch module not found, skipping torch.cuda.empty_cache()."
                )
            except Exception as e:
                logger.warning(f"Error during torch.cuda.empty_cache(): {e}")

    start_time = time.time()
    results, total_settings_tested = run_diversification_loop(
        rankings=rankings,
        pos_items=pos_items,
        embedder=embedder,
        experiment_params=experiment_params,
        baseline_metrics=baseline_metrics,
        categories_data=categories_data,
    )
    end_time = time.time()
    total_duration = end_time - start_time

    if total_settings_tested > 0:
        avg_time_per_setting = total_duration / total_settings_tested
        logger.info(f"Total parameter settings tested: {total_settings_tested}")
        logger.info(
            f"Average time per parameter setting: {avg_time_per_setting:.2f} seconds"
        )
    else:
        logger.info("No parameter settings were tested.")

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

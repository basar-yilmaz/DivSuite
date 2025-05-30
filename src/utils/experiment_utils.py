"""Utilities for experiment setup and configuration."""

import datetime
import os

from src.core.embedders.ste_embedder import STEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_experiment_params(config: dict) -> dict:
    """
    Extract and validate experiment parameters from config.

    Args:
        config (dict): Raw configuration dictionary.

    Returns:
        dict: Processed experiment parameters including:
            - diversifier_cls: The diversifier class to use
            - diversifier_param_name: Name of the parameter to vary
            - param_start: Starting value for parameter
            - param_end: Ending value for parameter
            - param_step: Step size for parameter variation
            - threshold_drop: Maximum allowed NDCG drop
            - top_k: Number of top items to consider
            - use_category_ild: Whether to use category-based ILD
    """
    diversifier_cls = _get_diversifier_class(config["experiment"]["diversifier"])

    return {
        "diversifier_cls": diversifier_cls,
        "diversifier_param_name": config["experiment"]["param_name"],
        "param_start": config["experiment"]["param_start"],
        "param_end": config["experiment"]["param_end"],
        "param_step": config["experiment"]["param_step"],
        "threshold_drop": config["experiment"]["threshold_drop"],
        "top_k": config["experiment"]["top_k"],
        "use_category_ild": config["experiment"].get("use_category_ild", False),
    }


def get_similarity_params(config: dict) -> dict:
    """
    Extract and validate similarity parameters from config.

    Args:
        config (dict): Raw configuration dictionary.

    Returns:
        dict: Processed similarity parameters including:
            - use_similarity_scores: Whether to use similarity scores
            - similarity_scores_path: Path to similarity scores file
    """
    return {
        "use_similarity_scores": config["similarity"]["use_similarity_scores"],
        "similarity_scores_path": config["similarity"]["similarity_scores_path"],
    }


def get_embedding_params(config: dict) -> dict:
    """
    Extract and validate embedding parameters from config.

    Args:
        config (dict): Raw configuration dictionary.

    Returns:
        dict: Processed embedding parameters including:
            - use_precomputed_embeddings: Whether to use precomputed embeddings
            - precomputed_embeddings_path: Path to the precomputed embeddings file
    """
    return {
        "use_precomputed_embeddings": config["embedder"]["use_precomputed_embeddings"],
        "precomputed_embeddings_path": config["embedder"][
            "precomputed_embeddings_path"
        ],
    }


def _get_diversifier_class(class_name: str) -> type:
    """
    Get diversifier class from its name using the algorithm registry.

    Args:
        class_name: Name of the diversifier class (case insensitive)
                   e.g., 'sy', 'SY', 'sydiversifier' all map to SYDiversifier

    Returns:
        Type: The diversifier class

    Raises:
        ValueError: If the algorithm is not registered or module cannot be imported
    """
    import importlib

    from src.utils.algorithm_registry import (
        get_algorithm_info,
        get_registered_algorithms,
        normalize_algorithm_name,
    )

    try:
        algorithm_info = get_algorithm_info(class_name)
        if not algorithm_info:
            available_algorithms = get_registered_algorithms()
            canonical_name = normalize_algorithm_name(class_name)
            error_msg = (
                f"Algorithm '{class_name}' not found in registry. "
                f"Available algorithms: {', '.join(available_algorithms)}"
            )
            if canonical_name:
                error_msg += f"\nDid you mean '{canonical_name}'?"
            logger.warning(error_msg)
            raise ValueError(
                f"Algorithm '{class_name}' not registered. Please add it to algorithm_registry.py first."
            )

        try:
            module = importlib.import_module(algorithm_info["module"])
            return getattr(module, algorithm_info["class"])
        except (ImportError, AttributeError) as e:
            logger.error(
                f"Failed to import {algorithm_info['class']} from {algorithm_info['module']}: {e!s}"
            )
            raise ValueError(
                f"Failed to load algorithm '{class_name}'. Please check if the module and class exist."
            ) from e

    except Exception as e:
        logger.error(f"Unexpected error loading algorithm '{class_name}': {e!s}")
        raise


def setup_results_directory(diversifier_cls: type) -> tuple[str, str]:
    """
    Create results directory and generate timestamp.

    Args:
        diversifier_cls: The diversifier class being evaluated.

    Returns:
        Tuple[str, str]: (results_folder_path, timestamp_string)
    """
    diversifier_name = diversifier_cls.__name__.replace("Diversifier", "").lower()
    results_folder = f"results_{diversifier_name}"
    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Created results directory: {results_folder}")
    return results_folder, timestamp


def initialize_embedder(config: dict, embedding_params: dict) -> STEmbedder:
    """
    Initialize the sentence transformer embedder.

    Args:
        config (dict): Configuration containing embedder settings.
        embedding_params (dict): Embedding parameters (use_precomputed_embeddings: bool, precomputed_embeddings_path: str).

    Returns:
        STEmbedder: Initialized embedder instance.
    """
    # If use_precomputed_embeddings is true, we don't need to initialize the embedder.
    if embedding_params["use_precomputed_embeddings"]:
        logger.info("Using precomputed embeddings. Embedder not initialized.")
        return None

    if config["similarity"]["use_similarity_scores"]:
        logger.info("Using similarity scores. Embedder not initialized.")
        return None

    # If use_precomputed_embeddings is false, we need to initialize the embedder.
    embedder = STEmbedder(
        model_name=config["embedder"]["model_name"],
        device=config["embedder"]["device"],
        batch_size=config["embedder"]["batch_size"],
    )
    logger.info(f"Initialized embedder with model: {config['embedder']['model_name']}")
    return embedder

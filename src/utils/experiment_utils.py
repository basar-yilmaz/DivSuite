"""Utilities for experiment setup and configuration."""

import os
import datetime
from typing import Tuple, Type

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


def _get_diversifier_class(class_name: str) -> Type:
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
        get_registered_algorithms,
        get_algorithm_info,
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
                f"Failed to import {algorithm_info['class']} from {algorithm_info['module']}: {str(e)}"
            )
            raise ValueError(
                f"Failed to load algorithm '{class_name}'. Please check if the module and class exist."
            ) from e

    except Exception as e:
        logger.error(f"Unexpected error loading algorithm '{class_name}': {str(e)}")
        raise


def setup_results_directory(diversifier_cls: Type) -> Tuple[str, str]:
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


def initialize_embedder(config: dict) -> STEmbedder:
    """
    Initialize the sentence transformer embedder.

    Args:
        config (dict): Configuration containing embedder settings.

    Returns:
        STEmbedder: Initialized embedder instance.
    """
    embedder = STEmbedder(
        model_name=config["embedder"]["model_name"],
        device=config["embedder"]["device"],
        batch_size=config["embedder"]["batch_size"],
    )
    logger.info(f"Initialized embedder with model: {config['embedder']['model_name']}")
    return embedder

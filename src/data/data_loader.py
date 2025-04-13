"""Module for loading and preparing experiment data."""

import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Any

from src.data.data_utils import load_data_and_convert
from src.data.category_utils import load_movie_categories
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_and_validate_path(
    base_path: Path, config_key: str, config_data: dict, is_dir: bool = False
) -> Path:
    """Helper function to resolve and validate a file or directory path from config."""
    path_str = config_data.get(config_key)
    if not path_str:
        raise ValueError(
            f"Configuration error: '{config_key}' is missing or empty in config['data']."
        )

    resolved_path = base_path / path_str

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configuration error: Path specified by '{config_key}' not found at '{resolved_path}'."
        )

    if is_dir and not resolved_path.is_dir():
        raise ValueError(
            f"Configuration error: Path specified by '{config_key}' is not a directory: '{resolved_path}'."
        )
    elif not is_dir and not resolved_path.is_file():
        raise ValueError(
            f"Configuration error: Path specified by '{config_key}' is not a file: '{resolved_path}'."
        )

    return resolved_path


def load_experiment_data(
    config: dict, use_category_ild: bool
) -> Tuple[Dict[int, Tuple[List[str], List[float]]], List[str], Any, Dict[Any, str]]:
    """
    Load and prepare all data required for the experiment.

    Args:
        config (dict): Configuration containing data paths and settings.
        use_category_ild (bool): Whether to load category/genre data.

    Returns:
        Tuple containing:
            - rankings (Dict[int, Tuple[List[str], List[float]]]): User rankings.
            - pos_items (List[str]): List of positive items for each user.
            - categories_data (Any): Category information or None.
            - id_to_title (Dict[Any, str]): Mapping from item ID to item title/name.
    """
    data_config = config["data"]
    base_path = Path(data_config["base_path"])

    # Validate base path
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(
            f"Configuration error: Base data path not found or not a directory: '{base_path}'"
        )

    # Load category data if needed
    categories_data = None
    if use_category_ild:
        categories_path = _resolve_and_validate_path(
            base_path, "movie_categories", data_config
        )
        categories_data = load_movie_categories(str(categories_path))
        logger.info(f"Loaded category data from {categories_path}")

    # Load main experiment data (resolve and validate paths)
    item_mappings_path = _resolve_and_validate_path(
        base_path, "item_mappings", data_config
    )
    test_samples_path = _resolve_and_validate_path(
        base_path, "test_samples", data_config
    )

    converted_data, id_to_title = load_data_and_convert(
        str(test_samples_path), str(item_mappings_path)
    )
    pos_items = [row[0] for row in converted_data]
    logger.info(
        f"Loaded and converted test samples data from {test_samples_path} and {item_mappings_path}"
    )

    # Load rankings data (resolve and validate paths)
    topk_list_path = _resolve_and_validate_path(base_path, "topk_list", data_config)
    topk_scores_path = _resolve_and_validate_path(base_path, "topk_scores", data_config)

    try:
        with open(topk_list_path, "rb") as f:
            topk_list = pickle.load(f)
        with open(topk_scores_path, "rb") as f:
            topk_scores = pickle.load(f)
        logger.info(f"Loaded ranking data from {topk_list_path} and {topk_scores_path}")
    except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
        logger.error(f"Error loading ranking data: {e}")
        raise RuntimeError(
            f"Failed to load ranking data from {topk_list_path} or {topk_scores_path}"
        ) from e

    # Build rankings dictionary
    rankings = {}
    if len(converted_data) != len(topk_list) or len(converted_data) != len(topk_scores):
        logger.warning(
            f"Mismatch in lengths: converted_data ({len(converted_data)}), topk_list ({len(topk_list)}), topk_scores ({len(topk_scores)}). Rankings might be incomplete."
        )
        # Decide how to handle mismatch, e.g., use min length or raise error
        min_len = min(len(converted_data), len(topk_list), len(topk_scores))
    else:
        min_len = len(converted_data)

    for idx in range(min_len):
        items, scores = topk_list[idx], topk_scores[idx]
        # Ensure the items retrieved from topk_list are valid indices for converted_data
        try:
            # Assuming items in topk_list are indices/keys into converted_data[idx] (needs verification based on actual data structure)
            # If items are actual titles/IDs, this logic needs adjustment
            # Example assumes items are indices:
            # titles = [converted_data[idx][item_index] for item_index in items] # Adjust if 'items' are not indices
            # This part depends heavily on the structure of topk_list and converted_data
            # Using the original logic for now, assuming items are indices to titles in converted_data[idx]
            titles = [converted_data[idx][item] for item in items]  # Original logic
            rankings[idx + 1] = (titles, scores.tolist())
        except IndexError:
            logger.error(
                f"IndexError while building rankings for user index {idx}. Skipping user."
            )
            continue
        except Exception as e:
            logger.error(
                f"Unexpected error building rankings for user index {idx}: {e}. Skipping user."
            )
            continue

    # rankings = { # Original implementation - keep for reference if needed
    #     idx + 1: ([converted_data[idx][item] for item in items], scores.tolist())
    #     for idx, (items, scores) in enumerate(zip(topk_list, topk_scores))
    # }
    logger.info(f"Built rankings dictionary for {len(rankings)} users")

    return rankings, pos_items, categories_data, id_to_title

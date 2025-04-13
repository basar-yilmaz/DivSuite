"""Module for loading and preparing experiment data."""

import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Any
import os

from src.data.data_utils import load_data_and_convert
from src.data.category_utils import load_movie_categories
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_experiment_data(
    config: dict, use_category_ild: bool
) -> Tuple[Dict[int, Tuple[List[str], List[float]]], List[str], Any]:
    """
    Load and prepare all data required for the experiment.

    Args:
        config (dict): Configuration containing data paths and settings.
        use_category_ild (bool): Whether to load category/genre data.

    Returns:
        Tuple containing:
            - rankings (Dict[int, Tuple[List[str], List[float]]]):
                User rankings mapping user IDs to (titles, scores).
            - pos_items (List[str]): List of positive items for each user.
            - categories_data (Any): Category information if use_category_ild is True,
                                   None otherwise.
    """
    data_path = Path(config["data"]["base_path"])

    # Load category data if needed
    categories_data = None
    if use_category_ild:
        category_path_str = config["data"].get("movie_categories")
        if not category_path_str:
            raise ValueError(
                "Configuration error: 'use_category_ild' is true, but 'movie_categories' path is missing or empty in config['data']."
            )

        categories_path = data_path / category_path_str

        if not os.path.exists(categories_path):
            raise FileNotFoundError(
                f"Configuration error: Movie categories file not found at '{categories_path}'. "
                f"'use_category_ild' is true, but the specified file does not exist."
            )
        if not os.path.isfile(categories_path):
            raise ValueError(
                f"Configuration error: Path specified for 'movie_categories' is not a file: '{categories_path}'."
            )

        categories_data = load_movie_categories(str(categories_path))
        logger.info("Loaded category data")

    # Load main experiment data
    item_mappings = data_path / config["data"]["item_mappings"]
    test_samples = data_path / config["data"]["test_samples"]
    converted_data, id_to_title = load_data_and_convert(
        str(test_samples), str(item_mappings)
    )
    pos_items = [row[0] for row in converted_data]
    logger.info("Loaded and converted test samples data")

    # Load rankings data
    topk_list_path = data_path / config["data"]["topk_list"]
    topk_scores_path = data_path / config["data"]["topk_scores"]

    with open(topk_list_path, "rb") as f:
        topk_list = pickle.load(f)
    with open(topk_scores_path, "rb") as f:
        topk_scores = pickle.load(f)
    logger.info("Loaded ranking data")

    # Build rankings dictionary
    rankings = {
        idx + 1: ([converted_data[idx][item] for item in items], scores.tolist())
        for idx, (items, scores) in enumerate(zip(topk_list, topk_scores))
    }
    logger.info(f"Built rankings dictionary for {len(rankings)} users")

    return rankings, pos_items, categories_data, id_to_title

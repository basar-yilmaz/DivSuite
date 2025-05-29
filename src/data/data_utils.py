"""Utilities for loading and processing data."""

import ast
import contextlib
import csv


def load_data_and_convert(data_csv_path: str, mapping_csv_path: str) -> list:
    """
    Load user interaction data and convert item IDs to titles.

    Args:
        data_csv_path (str): Path to CSV file with columns "user_id", "pos_item", "neg_items".
        mapping_csv_path (str): Path to CSV file with columns "item_id" and "item" (title).

    Returns:
        list: A list of lists, where each inner list contains:
             [pos_item_title, neg_item_title1, neg_item_title2, ...]
    """
    id_to_title = _load_id_to_title_mapping(mapping_csv_path)
    return _process_data_file(data_csv_path, id_to_title), id_to_title


def _load_id_to_title_mapping(mapping_csv_path: str) -> dict:
    """Load the ID to title mapping from CSV file."""
    id_to_title = {}
    with open(mapping_csv_path, encoding="utf-8") as map_file:
        reader = csv.DictReader(map_file)
        for row in reader:
            try:
                key = int(row["item_id"])
            except ValueError:
                key = row["item_id"]
            id_to_title[key] = row["item"]
    return id_to_title


def _process_data_file(data_csv_path: str, id_to_title: dict) -> list:
    """Process the data file and convert IDs to titles."""
    results = []
    with open(data_csv_path, encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            pos_item_title = _convert_id_to_title(row["pos_item"], id_to_title)
            neg_items_titles = _process_negative_items(row["neg_items"], id_to_title)
            results.append([pos_item_title, *neg_items_titles])
    return results


def _convert_id_to_title(item_id: str, id_to_title: dict) -> str:
    """Convert a single item ID to its title."""
    with contextlib.suppress(ValueError):
        item_id = int(item_id)
    return id_to_title.get(item_id, str(item_id))


def _process_negative_items(neg_items_str: str, id_to_title: dict) -> list:
    """Process negative items string and convert to titles."""
    try:
        neg_items_ids = ast.literal_eval(neg_items_str)
    except Exception as e:
        print(f"Error parsing neg_items: {neg_items_str}. Error: {e}")
        return []

    return [_convert_id_to_title(item, id_to_title) for item in neg_items_ids]


def create_relevance_lists(data: dict, pos_items: list) -> list:
    """
    Create binary relevance lists for each user's recommendations.

    Args:
        data (dict): Dictionary mapping user IDs to tuples of (titles, relevance_scores).
        pos_items (list): List of positive items where pos_items[user_id-1] is the
                         positive item title for user_id.

    Returns:
        list: List of binary relevance lists where 1 indicates the positive item
              and 0 indicates other items.
    """
    relevance_lists = []
    for user_id, (titles, _) in data.items():
        relevance_list = [0] * len(titles)
        pos_index = get_positive_index(titles, pos_items[user_id - 1])
        if pos_index != -1:
            relevance_list[pos_index] = 1
        relevance_lists.append(relevance_list)
    return relevance_lists


def get_positive_index(data: list, item_title: str) -> int:
    """
    Get the index of a positive item in a list.

    Args:
        data (list): List of item titles.
        item_title (str): Title to find.

    Returns:
        int: Index of the item if found, -1 otherwise.
    """
    try:
        return data.index(item_title)
    except ValueError:
        return -1

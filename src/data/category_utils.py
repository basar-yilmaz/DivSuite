"""Utilities for handling movie categories and genres."""

import csv

import numpy as np

GENRES = [
    "Drama",
    "Comedy",
    "Action",
    "Thriller",
    "Romance",
    "Adventure",
    "Children's",
    "Crime",
    "Sci-Fi",
    "Horror",
    "War",
    "Mystery",
    "Musical",
    "Documentary",
    "Animation",
    "Western",
    "Film-Noir",
    "Fantasy",
    "unknown",
]


def create_genre_vector(categories: set) -> np.ndarray:
    """
    Create a binary genre vector from a set of categories.

    Args:
        categories (set): Set of genre categories for a movie.

    Returns:
        np.ndarray: Binary vector where 1 indicates presence of genre.
    """
    vector = np.zeros(len(GENRES))
    for i, genre in enumerate(GENRES):
        if genre in categories:
            vector[i] = 1
    return vector


def load_movie_categories(mapping_csv_path: str) -> tuple:
    """
    Load movie categories from CSV file.

    Args:
        mapping_csv_path (str): Path to CSV file with columns "item_id" and "class".

    Returns:
        tuple: (title_to_categories, title_to_vector) where:
               - title_to_categories: Dict mapping titles to sets of categories
               - title_to_vector: Dict mapping titles to binary genre vectors
    """
    title_to_categories = {}
    title_to_vector = {}

    with open(mapping_csv_path, encoding="utf-8") as map_file:
        reader = csv.DictReader(map_file)
        for row in reader:
            title = row["item_id"]
            categories = set(row["class"].split())
            title_to_categories[title] = categories
            title_to_vector[title] = create_genre_vector(categories)

    return title_to_categories, title_to_vector

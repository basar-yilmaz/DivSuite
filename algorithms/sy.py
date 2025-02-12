import numpy as np
from algorithms.base import BaseDiversifier
from utils import compute_pairwise_cosine
from embedders.base_embedder import BaseEmbedder


class SYDiversifier(BaseDiversifier):
    """
    Implements the SY diversification algorithm based on the Groundhog Day paper.
    This algorithm eliminates items that is too similar to the previous items incrementally.
    """

    def __init__(self, embedder: BaseEmbedder, threshold: float = 0.67):
        """
        :param embedder: An embedder instance (e.g. STEmbedder or HFEmbedder)
        :param threshold: Similarity threshold.
        """
        self.threshold = threshold
        self.embedder = embedder

    def diversify(self, items: np.ndarray, top_k: int = 10, **kwargs) -> np.ndarray:
        """
        Diversify the given array of items using the SY algorithm.

        :param items: A 2D array of shape (N, 3), e.g., [id, title, relevance_score].
        :param top_k: Number of items to retrieve after diversification.
        :param kwargs: Additional parameters (unused in this implementation).
        :return: A 2D array of shape (top_k, 3), diversified.
        """
        num_items = items.shape[0]
        if num_items == 0:
            return items

        # Ensure top_k is within bounds
        top_k = min(top_k, num_items)

        # Sort items by descending relevance
        sorted_idx = np.argsort(items[:, 2].astype(float))[::-1]
        items = items[sorted_idx]

        # Extract titles to embed
        titles = items[:, 1].tolist()
        embeddings = self.embedder.encode_batch(titles)
        sim_matrix = compute_pairwise_cosine(embeddings)

        selected_indices = [0]

        # Iterate through each item to check for similarity with selected items
        for i in range(1, num_items):
            # Assume the item is selected
            is_selected = True

            # Check similarity with all previously selected items
            for j in selected_indices:
                if sim_matrix[i, j] > self.threshold:
                    is_selected = False
                    break

            if is_selected:
                selected_indices.append(i)

            # Stop if we've reached top_k
            if len(selected_indices) == top_k:
                break

        return items[selected_indices]

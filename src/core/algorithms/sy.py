import numpy as np

from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class SYDiversifier(BaseDiversifier):
    """
    Implements the SY diversification algorithm based on the Groundhog Day paper.
    This algorithm eliminates items that is too similar to the previous items incrementally.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.67,
        use_similarity_scores: bool = False,
        item_id_mapping: dict = None,
        similarity_scores_path: str = None,
    ):
        """
        :param embedder: An embedder instance (e.g. STEmbedder or HFEmbedder)
        :param threshold: Similarity threshold.
        """
        super().__init__(
            embedder=embedder,
            use_similarity_scores=use_similarity_scores,
            item_id_mapping=item_id_mapping,
            similarity_scores_path=similarity_scores_path,
        )
        self.threshold = threshold

    def diversify(
        self, items: np.ndarray, top_k: int = 10, title2embedding: dict = None, **kwargs
    ) -> np.ndarray:
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

        # Calculate similarity matrix using the base class method
        titles = items[:, 1].tolist()
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        selected_indices = [0]

        # Iterate through each item to check for similarity with selected items
        for i in range(1, num_items):
            # Vectorized check: if any selected item has similarity above threshold, skip
            if not np.any(sim_matrix[i, selected_indices] > self.threshold):
                selected_indices.append(i)

            # Stop if we've reached top_k
            if len(selected_indices) == top_k:
                break

        return items[selected_indices]

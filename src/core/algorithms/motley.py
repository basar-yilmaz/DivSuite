import numpy as np

from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class MotleyDiversifier(BaseDiversifier):
    """
    Implements the Motley algorithm (Algorithm 5), which:
      1) Picks the highest-relevance item s_s from S, moves it to R
      2) While |R| < k:
           - pick s_s from S with the highest relevance
           - if for all s_r in R, div(s_r, s_s) >= threshold => R <- R + { s_s }
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        theta_: float = 0.5,
        use_similarity_scores: bool = False,
        item_id_mapping: dict = None,
        similarity_scores_path: str = None,
    ):
        super().__init__(
            embedder=embedder,
            use_similarity_scores=use_similarity_scores,
            item_id_mapping=item_id_mapping,
            similarity_scores_path=similarity_scores_path,
        )
        self.theta_ = theta_

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance]
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array: [item_id, title, relevance_score]
        :param top_k: number of items we want in R
        :return: A subset of items, up to size k, that meets the Motley criterion.
        """

        num_items = items.shape[0]
        if num_items == 0:
            return items

        # If fewer items than k, just clamp top_k
        top_k = min(top_k, num_items)

        # Calculate similarity matrix using the base class method
        titles = items[:, 1].tolist()
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        # Pre-sort indices by descending relevance
        sorted_indices = sorted(range(num_items), key=lambda i: float(items[i, 2]), reverse=True)
        R = []

        # Iterate through the sorted indices and build R
        for idx in sorted_indices:
            if not R:
                R.append(idx)
            else:
                # Vectorized diversity check:
                # For each candidate, compute distances to all items in R in one go.
                if np.all(1.0 - sim_matrix[np.array(R), idx] >= self.theta_):
                    R.append(idx)
            if len(R) == top_k:
                break


        return items[R]

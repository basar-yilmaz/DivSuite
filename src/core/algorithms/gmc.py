import numpy as np

from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class GMCDiversifier(BaseDiversifier):
    """
    Implements the Greedy Marginal Contribution (GMC) diversification algorithm.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        lambda_: float = 0.5,
        use_similarity_scores: bool = False,
        item_id_mapping: dict | None = None,
        similarity_scores_path: str | None = None,
    ):
        super().__init__(
            embedder=embedder,
            use_similarity_scores=use_similarity_scores,
            item_id_mapping=item_id_mapping,
            similarity_scores_path=similarity_scores_path,
        )
        self.lambda_ = lambda_

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance]
        top_k: int = 10,
        title2embedding: dict | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array, e.g. [item_id, title, relevance_score]
        :param top_k: desired size of the final diversified set R
        :param lambda_: trade-off parameter between relevance and diversity
        :return: A subset of 'items' of size top_k (or fewer if items < top_k)
        """

        num_items = items.shape[0]
        if num_items == 0:
            return items

        # Clamp top_k if necessary
        top_k = min(top_k, num_items)

        # Calculate similarity matrix using the base class method
        titles = items[:, 1].tolist()  # item text
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        # Initialize empty result set
        R = []
        S = set(range(num_items))  # Candidate pool

        while len(R) < top_k and len(S) > 0:
            best_s = max(S, key=lambda s: self.mmc(s, R, S, items, sim_matrix, top_k))
            S.remove(best_s)
            R.append(best_s)

        return items[R]

    def mmc(self, s_i, R, S, items, sim_matrix, k):
        """
        Computes the Maximum Marginal Contribution (MMC) score for an item.
        """
        if not R:
            return (1 - self.lambda_) * items[s_i, 2]

        p = len(R) + 1  # When s_i is added, R’s size increases by 1
        k_minus_1 = max(1, k - 1)

        # 1. Relevance contribution
        sim_term = (1 - self.lambda_) * items[s_i, 2]

        # 2. Diversity contribution from R (vectorized over R)
        R_arr = np.array(list(R))
        div_scores_R = 1.0 - sim_matrix[s_i, R_arr]
        div_term_R = (self.lambda_ / k_minus_1) * np.sum(div_scores_R)

        # 3. Diversity contribution from S:
        remaining_depth = k - p  # How many additional diversity terms to consider
        div_term_S = 0.0
        if remaining_depth > 0:
            # Convert S to array and exclude s_i
            S_arr = np.array(list(S))
            S_arr = S_arr[S_arr != s_i]
            div_scores_S = 1.0 - sim_matrix[s_i, S_arr]

            # If we have more candidates than needed, pick the top remaining_depth contributions
            if div_scores_S.size > remaining_depth:
                # np.sort returns ascending order, so take the last 'remaining_depth' elements for descending order
                top_l_scores = np.sort(div_scores_S)[-remaining_depth:]
            else:
                top_l_scores = div_scores_S

            div_term_S = (self.lambda_ / k_minus_1) * np.sum(top_l_scores)

        return sim_term + div_term_R + div_term_S

import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
from src.core.embedders.base_embedder import BaseEmbedder


class GNEDiversifier(BaseDiversifier):
    """
    Implements the Greedy Randomized with Neighborhood Expansion (GNE)
    diversification algorithm using a GRASP framework.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        lambda_: float = 0.5,
        alpha: float = 0.01,
        imax: int = 5,
    ):
        self.lambda_ = lambda_
        self.embedder = embedder
        self.alpha = alpha  # Controls the size of the Restricted Candidate List (RCL)
        self.imax = imax  # Number of GRASP iterations

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance_score]
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        num_items = items.shape[0]
        if num_items == 0:
            return items
        top_k = min(top_k, num_items)
        titles = items[:, 1].tolist()
        if title2embedding is not None:
            try:
                embeddings = np.stack([title2embedding[title] for title in titles])
            except KeyError as e:
                raise ValueError(f"Missing embedding for title: {e}")
        else:
            embeddings = self.embedder.encode_batch(titles)
        sim_matrix = compute_pairwise_cosine(embeddings)

        best_R = None
        best_F = -np.inf
        full_indices = set(range(num_items))
        for _ in range(self.imax):
            # --- Construction Phase ---
            R = []
            S = set(range(num_items))
            while len(R) < top_k and S:
                # Compute mmc scores for candidates in S
                mmc_scores = {
                    s: self._mmc(s, R, S, items, sim_matrix, top_k) for s in S
                }
                s_max = max(mmc_scores.values())
                s_min = min(mmc_scores.values())
                threshold = s_max - self.alpha * (s_max - s_min)
                RCL = [s for s in S if mmc_scores[s] >= threshold]
                # Randomly select from the RCL
                s_choice = np.random.choice(RCL)
                R.append(s_choice)
                S.remove(s_choice)

            # --- Local Search Phase ---
            R = self._local_search(R, full_indices, items, sim_matrix, top_k)
            current_F = self._compute_F(R, items, sim_matrix, top_k)
            if current_F > best_F:
                best_F = current_F
                best_R = R

        return items[best_R]

    def _mmc(self, s_i, R, S, items, sim_matrix, k):
        """
        Computes the Maximum Marginal Contribution (MMC) score for an item.
        (Same as GMC implementation)
        """
        if not R:
            return (1 - self.lambda_) * items[s_i, 2]
        p = len(R) + 1
        k_minus_1 = max(1, k - 1)
        sim_term = (1 - self.lambda_) * items[s_i, 2]
        div_sum_R = sum(1 - sim_matrix[s_i, s_j] for s_j in R)
        div_term_R = (self.lambda_ / k_minus_1) * div_sum_R
        remaining_depth = k - p
        div_term_S = 0
        if remaining_depth > 0:
            diversity_scores = [(1 - sim_matrix[s_i, s_j]) for s_j in S if s_j != s_i]
            top_l_scores = sorted(diversity_scores, reverse=True)[:remaining_depth]
            div_term_S = (self.lambda_ / k_minus_1) * sum(top_l_scores)
        return sim_term + div_term_R + div_term_S

    def _compute_F(self, R, items, sim_matrix, k):
        """
        Computes the overall objective function:
        F = (k - 1) * (1 - lambda) * (sum of relevance scores) + 2 * lambda * (sum of pairwise diversity)
        """
        sim_sum = sum(items[s, 2] for s in R)
        div_sum = sum(
            1 - sim_matrix[s_i, s_j] for i, s_i in enumerate(R) for s_j in R[i + 1 :]
        )
        return (k - 1) * (1 - self.lambda_) * sim_sum + 2 * self.lambda_ * div_sum

    def _local_search(self, R, full_indices, items, sim_matrix, k):
        """
        Improves the solution R by trying to swap its elements with candidates not in R.
        """
        improved = True
        current_R = R.copy()
        S_remaining = list(full_indices - set(current_R))
        while improved:
            improved = False
            current_F = self._compute_F(current_R, items, sim_matrix, k)
            for i in range(len(current_R)):
                for candidate in S_remaining:
                    new_R = current_R.copy()
                    new_R[i] = candidate
                    new_F = self._compute_F(new_R, items, sim_matrix, k)
                    if new_F > current_F:
                        # Swap: update solution and candidate pool
                        S_remaining.append(current_R[i])
                        current_R[i] = candidate
                        S_remaining.remove(candidate)
                        improved = True
                        current_F = new_F
                        break
                if improved:
                    break
        return current_R

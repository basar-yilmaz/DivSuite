import numpy as np

from src.core.algorithms.base import BaseDiversifier
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
        alpha: float = 0,
        imax: int = 10,
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
        self.alpha = alpha  # Controls the size of the Restricted Candidate List (RCL)
        self.imax = imax  # Number of GRASP iterations

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance_score]
        top_k: int = 10,
        title2embedding: dict | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Optimized diversify method with faster embedding handling."""
        num_items = items.shape[0]
        if num_items == 0:
            return items

        top_k = min(top_k, num_items)

        # Compute similarity matrix using the base class method
        titles = items[:, 1]
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        # Precompute diversity matrix (1 - similarity) for efficiency
        div_matrix = 1 - sim_matrix

        # The rest of the code remains structurally the same
        # but benefits from the optimized helper methods above
        best_R = None
        best_F = -np.inf
        full_indices = set(range(num_items))

        for _ in range(self.imax):
            # Construction Phase
            R = []
            S = set(range(num_items))

            while len(R) < top_k and S:
                mmc_scores = {
                    s: self._mmc(s, R, S, items, div_matrix, top_k) for s in S
                }
                s_max = max(mmc_scores.values())
                s_min = min(mmc_scores.values())
                threshold = s_max - self.alpha * (s_max - s_min)
                RCL = [s for s in S if mmc_scores[s] >= threshold]
                s_choice = np.random.choice(RCL)
                R.append(s_choice)
                S.remove(s_choice)

            # Local Search Phase
            R = self._local_search(R, full_indices, items, div_matrix, top_k)
            current_F = self._compute_F(R, items, div_matrix, top_k)

            if current_F > best_F:
                best_F = current_F
                best_R = R

        return items[best_R]

    def _mmc(self, s_i, R, S, items, div_matrix, k):
        """
        Computes the MMC score for an item using vectorized operations.
        """
        k_minus_1 = max(1, k - 1)
        # Relevance contribution remains as before.
        sim_term = (1 - self.lambda_) * items[s_i, 2]

        if not R:
            return sim_term

        # Vectorize diversity contribution from R.
        R_arr = np.array(list(R))
        div_term_R = (self.lambda_ / k_minus_1) * np.sum(div_matrix[s_i, R_arr])

        # Diversity contribution from S.
        remaining_depth = k - len(R) - 1
        if remaining_depth <= 0:
            return sim_term + div_term_R

        # Vectorize over S: exclude s_i.
        S_arr = np.array(list(S))
        S_arr = S_arr[S_arr != s_i]
        if S_arr.size > 0:
            diversity_scores = div_matrix[s_i, S_arr]
            # Use np.partition to efficiently get the top 'remaining_depth' scores.
            top_l_scores = np.partition(
                diversity_scores, -min(remaining_depth, diversity_scores.size)
            )[-remaining_depth:]
            div_term_S = (self.lambda_ / k_minus_1) * np.sum(top_l_scores)
            return sim_term + div_term_R + div_term_S

        return sim_term + div_term_R

    def _compute_F(self, R, items, div_matrix, k):
        """
        Computes the overall objective function with vectorized operations where possible.
        Uses precomputed diversity matrix.
        """
        R_array = np.array(R)
        sim_sum = np.sum(items[R_array, 2])

        # Create pairs for diversity calculation
        idx1, idx2 = np.triu_indices(len(R), k=1)
        if len(idx1) > 0:
            pairs_div = div_matrix[np.array(R)[idx1], np.array(R)[idx2]]
            div_sum = np.sum(pairs_div)
        else:
            div_sum = 0

        return (k - 1) * (1 - self.lambda_) * sim_sum + 2 * self.lambda_ * div_sum

    def _local_search(self, R, full_indices, items, div_matrix, k):
        """
        Optimized local search using vectorized incremental evaluation for swap candidates.
        """
        current_R = R.copy()
        S_remaining = list(full_indices - set(current_R))
        current_F = self._compute_F(current_R, items, div_matrix, k)
        improved = True

        while improved:
            improved = False
            # Try to improve each element of current_R
            for i in range(len(current_R)):
                original = current_R[i]
                R_without = np.delete(
                    np.array(current_R), i
                )  # R excluding the current index

                candidates = np.array(S_remaining)
                if candidates.size == 0:
                    continue

                # Compute the relevance delta for all candidates at once.
                rel_diff = (
                    (k - 1)
                    * (1 - self.lambda_)
                    * (items[candidates, 2] - items[original, 2])
                )

                # Compute diversity difference for each candidate.
                # For each candidate c, compute sum_{r in R_without} div_matrix[c, r]
                diversity_candidate = np.sum(
                    div_matrix[candidates][:, R_without], axis=1
                )
                diversity_original = np.sum(div_matrix[original, R_without])
                delta_div = (
                    2 * self.lambda_ * (diversity_candidate - diversity_original)
                )

                # Total delta change for replacing original with each candidate.
                delta_F = rel_diff + delta_div

                # Select the candidate with the maximum delta.
                best_idx = np.argmax(delta_F)
                if delta_F[best_idx] > 0:
                    best_candidate = int(candidates[best_idx])
                    # Update S_remaining: remove best_candidate and add original.
                    S_remaining.remove(best_candidate)
                    if original not in S_remaining:
                        S_remaining.append(original)
                    current_R[i] = best_candidate
                    current_F += delta_F[best_idx]
                    improved = True
                    break  # Restart the search after a successful swap

            # If an improvement was made, continue the outer loop; otherwise, exit.
        return current_R

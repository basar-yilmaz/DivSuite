import numpy as np

from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class SwapDiversifier(BaseDiversifier):
    """
    Implements the 'Swap' diversification algorithm, based on the provided pseudocode.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        lambda_: float = 0.5,
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
        self.lambda_ = lambda_

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance]
        top_k: int = 10,
        title2embedding: dict = None,
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

        # Compute similarity matrix using the base class method
        titles = items[:, 1].tolist()  # item text
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        # Helper function to compute F(q, R') for a given set of indices
        def score_set(R_indices) -> float:
            if not R_indices:
                return 0.0

            # Convert to list so we can index by [i]
            R_list = list(R_indices)
            k = len(R_list)

            # Relevance part
            sum_relevance = float(np.sum(items[R_list, 2].astype(float)))
            relevance_term = (k - 1) * (1 - self.lambda_) * sum_relevance

            # Diversity part: sum of (1 - sim) for all pairs
            div_sum = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    div_sum += 1.0 - sim_matrix[R_list[i], R_list[j]]

            diversity_term = 2.0 * self.lambda_ * div_sum
            return relevance_term + diversity_term

        # --- PHASE 1 ---
        sorted_by_rel = sorted(range(num_items), key=lambda i: items[i, 2], reverse=True)
        R_indices = sorted_by_rel[:top_k]
        S_indices = sorted_by_rel[top_k:] # Keep these sorted by relevance

        R_set = set(R_indices)
        current_score = score_set(R_set) # Calculate initial score once O(k^2)

        # --- PHASE 2: Swapping with Delta Calculation ---
        for s_s in S_indices: # Iterate through remaining items in relevance order
            best_swap_sj = -1
            best_delta = 0.0 # We only swap if delta > 0

            # Try swapping s_s with each s_j in the current R_set
            for s_j in R_set:
                # Compute the diversity sum for s_s excluding s_j:
                sum_one_minus_sim_ss_R = 0.0
                for r_idx in R_set:
                    if r_idx != s_j:
                        sum_one_minus_sim_ss_R += (1.0 - sim_matrix[s_s, r_idx])
                
                # Compute s_j's diversity sum excluding itself (same as before)
                sum_one_minus_sim_sj_R = 0.0
                for r_idx in R_set:
                    if r_idx != s_j:
                        sum_one_minus_sim_sj_R += (1.0 - sim_matrix[s_j, r_idx])

                # Calculate Relevance Delta
                delta_rel = (top_k - 1) * (1 - self.lambda_) * (float(items[s_s, 2]) - float(items[s_j, 2]))

                # Calculate Diversity Delta
                # Gain terms with s_s, Lose terms with s_j
                # Need sum(1-sim(s_s, r)) for r in R-{s_j}
                # Need sum(1-sim(s_j, r)) for r in R-{s_j}
                # Note: sum_one_minus_sim_ss_R already calculated is sum(1-sim(s_s, r)) for r in R
                # Note: sum_one_minus_sim_sj_R calculated above is sum(1-sim(s_j, r)) for r in R-{s_j}
                delta_div = 2.0 * self.lambda_ * (sum_one_minus_sim_ss_R - sum_one_minus_sim_sj_R)

                # Total Delta for swapping s_j with s_s
                current_swap_delta = delta_rel + delta_div

                if current_swap_delta > best_delta:
                    best_delta = current_swap_delta
                    best_swap_sj = s_j

            # If the best swap improves the score (delta > 0), perform the swap
            if best_swap_sj != -1:
                R_set.remove(best_swap_sj)
                R_set.add(s_s)
                current_score += best_delta 

        final_indices = list(R_set)

        return items[final_indices]

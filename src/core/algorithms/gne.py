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
        alpha: float = 0,
        imax: int = 10,
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
        self.alpha = alpha  # Controls the size of the Restricted Candidate List (RCL)
        self.imax = imax  # Number of GRASP iterations

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance_score]
        top_k: int = 10,
        title2embedding: dict = None,
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
                mmc_scores = {s: self._mmc(s, R, S, items, sim_matrix, div_matrix, top_k) for s in S}
                s_max = max(mmc_scores.values())
                s_min = min(mmc_scores.values())
                threshold = s_max - self.alpha * (s_max - s_min)
                RCL = [s for s in S if mmc_scores[s] >= threshold]
                s_choice = np.random.choice(RCL)
                R.append(s_choice)
                S.remove(s_choice)
            
            # Local Search Phase
            R = self._local_search(R, full_indices, items, sim_matrix, div_matrix, top_k)
            current_F = self._compute_F(R, items, sim_matrix, div_matrix, top_k)
            
            if current_F > best_F:
                best_F = current_F
                best_R = R
        
        return items[best_R]

    def _mmc(self, s_i, R, S, items, sim_matrix, div_matrix, k):
        """
        Computes the Maximum Marginal Contribution (MMC) score for an item.
        Optimized version with precomputed diversity matrix.
        """
        k_minus_1 = max(1, k - 1)
        sim_term = (1 - self.lambda_) * items[s_i, 2]
        
        if not R:
            return sim_term
        
        # Only compute the diversity once for this item
        diversity_factor = self.lambda_ / k_minus_1
        
        # Use precomputed diversity matrix instead of calculating 1-sim every time
        div_sum_R = sum(div_matrix[s_i, s_j] for s_j in R)
        div_term_R = diversity_factor * div_sum_R
        
        # Check if we need to consider remaining items
        remaining_depth = k - len(R) - 1
        if remaining_depth <= 0:
            return sim_term + div_term_R
        
        # Use numpy operations for faster selection of top diversity scores
        remaining_indices = np.array([s_j for s_j in S if s_j != s_i])
        if len(remaining_indices) > 0:
            diversity_scores = div_matrix[s_i, remaining_indices]
            top_l_scores = np.partition(diversity_scores, -min(remaining_depth, len(diversity_scores)))[-remaining_depth:]
            div_term_S = diversity_factor * np.sum(top_l_scores)
            return sim_term + div_term_R + div_term_S
        
        return sim_term + div_term_R

    def _compute_F(self, R, items, sim_matrix, div_matrix, k):
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

    def _local_search(self, R, full_indices, items, sim_matrix, div_matrix, k):
        """
        Improved local search with fewer redundant calculations.
        Uses precomputed diversity matrix.
        """
        improved = True
        current_R = R.copy()
        S_remaining = list(full_indices - set(current_R))
        current_F = self._compute_F(current_R, items, sim_matrix, div_matrix, k)
        
        while improved:
            improved = False
            
            # Precompute swapping effects for all pairs in one iteration
            for i in range(len(current_R)):
                # Store original item to restore later if needed
                original_item = current_R[i]
                
                for j, candidate in enumerate(S_remaining):
                    # Temporarily swap
                    current_R[i] = candidate
                    new_F = self._compute_F(current_R, items, sim_matrix, div_matrix, k)
                    
                    if new_F > current_F:
                        # Keep this improvement
                        S_remaining[j] = original_item  # Update remaining candidates
                        S_remaining.remove(original_item)  # Remove swapped-out item from remaining list
                        S_remaining.append(original_item)  # Add back replaced item 
                        improved = True
                        current_F = new_F
                        break
                    else:
                        # Restore the item if no improvement
                        current_R[i] = original_item
                    
                if improved:
                    break
                
        return current_R

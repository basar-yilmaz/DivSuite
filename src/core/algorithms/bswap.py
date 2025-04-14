import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class BSwapDiversifier(BaseDiversifier):
    """
    Implements the BSWAP algorithm (Algorithm 3).

    Steps (roughly):
      1) R <- empty
      2) While |R| < k: pick s_s in S with the highest relevance, move from S to R
      3) s_d <- argmin_{si in R} (divContribution(R, si))
      4) s_s <- argmax_{si in S} (relevance)
      5) while [deltaSim(q, s_d) - deltaSim(q, s_s) <= theta] and |S|>0:
          if div( (R \ {s_d}) U {s_s} ) > div(R):
             R <- (R \ {s_d}) U {s_s}
             s_d <- argmin_{si in R} (divContribution(R, si))
          s_s <- argmax_{si in S} (relevance)
          S <- S \ {s_s}
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        theta_: float = 0.1,
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
        self.theta = theta_

    def diversify(
        self,
        items: np.ndarray,
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        num_items = items.shape[0]
        if num_items == 0:
            return items
        top_k = min(top_k, num_items)

        titles = items[:, 1].tolist()
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        def relevance(i: int) -> float:
            return float(items[i, 2])

        # --- Pre-computation & Initial Greedy Phase ---
        all_indices = list(range(num_items))
        all_indices.sort(key=lambda i: relevance(i), reverse=True)

        R_set = set(all_indices[:top_k])
        # S_indices sorted by relevance descending
        S_indices = all_indices[top_k:]

        if len(R_set) < 1 or len(S_indices) < 1:
            final_indices = list(R_set)
            # final_indices.sort(key=lambda i: relevance(i), reverse=True) # Optional sort
            return items[final_indices] if final_indices else np.array([])

        # --- Helper for Diversity Contribution (O(k)) ---
        def get_div_contribution(target_set, item_idx, sim_matrix):
            contrib = 0.0
            for j in target_set:
                if item_idx != j:
                    # Ensure indices are valid for sim_matrix if sparse
                    contrib += 1.0 - sim_matrix[item_idx, j]
            return contrib

        # --- Store current contributions for O(k) updates ---
        # O(k^2) initial calculation
        div_contributions = {
            r: get_div_contribution(R_set, r, sim_matrix) for r in R_set
        }

        # --- Find initial s_d (O(k) after contributions are known) ---
        def find_s_d(current_R_set, current_contributions):
            if not current_R_set:
                return -1  # Should not happen if k >= 1
            # Find item in R with the minimum contribution
            min_contrib = float("inf")
            s_d_cand = -1
            for r_item in current_R_set:
                if current_contributions[r_item] < min_contrib:
                    min_contrib = current_contributions[r_item]
                    s_d_cand = r_item
                # Tie-breaking (optional): could add secondary sort key like relevance if needed
                # elif current_contributions[r_item] == min_contrib:
                # handle tie if necessary, e.g., pick lower relevance one?
            return s_d_cand
            # Alternatively: return min(current_R_set, key=lambda r: current_contributions[r])

        s_d = find_s_d(R_set, div_contributions)
        if s_d == -1:  # Handle empty R case if it arises
            return items[list(R_set)]

        # --- BSWAP Loop (Optimized) ---
        s_idx_in_S = 0  # Pointer to the next s_s in sorted S_indices
        while s_idx_in_S < len(S_indices):
            s_s = S_indices[s_idx_in_S]

            # Check loop condition
            if relevance(s_d) - relevance(s_s) > self.theta:
                break  # Relevance drop too large, stop swapping

            # --- Calculate Diversity Delta (O(k)) ---
            # Contribution s_s would have with elements currently in R (excluding s_d)
            contrib_ss_with_R_minus_sd = 0.0
            for r in R_set:
                if r != s_d:
                    contrib_ss_with_R_minus_sd += 1.0 - sim_matrix[s_s, r]

            # Contribution s_d currently has with elements in R (excluding s_d)
            # This is simply the stored div_contribution of s_d
            contrib_sd_with_R_minus_sd = div_contributions[s_d]

            delta_div = contrib_ss_with_R_minus_sd - contrib_sd_with_R_minus_sd

            # --- Perform Swap if beneficial ---
            if delta_div > 1e-9:  # Use tolerance for float comparison
                # Swap occurs
                R_set.remove(s_d)
                R_set.add(s_s)

                # --- Update contributions incrementally (O(k)) ---
                _ = div_contributions.pop(s_d)  # Remove old s_d

                # Update contributions of remaining items r in R
                for r in R_set:
                    if r != s_s:  # Don't update the newly added item yet
                        # remove effect of s_d, add effect of s_s
                        sim_r_sd = sim_matrix[r, s_d]
                        sim_r_ss = sim_matrix[r, s_s]
                        div_contributions[r] = (
                            div_contributions[r] - (1.0 - sim_r_sd) + (1.0 - sim_r_ss)
                        )

                # Calculate contribution for the new item s_s
                div_contributions[s_s] = get_div_contribution(R_set, s_s, sim_matrix)

                # Find the new s_d based on updated contributions (O(k))
                s_d = find_s_d(R_set, div_contributions)
                if s_d == -1:
                    break  # Should not happen

            # --- Move to next candidate s_s ---
            s_idx_in_S += 1
            # s_d remains the same if no swap occurred

        # --- Return final result ---
        final_indices = list(R_set)
        # Optional: Sort final list by relevance if desired, BSWAP doesn't guarantee order
        # final_indices.sort(key=lambda i: relevance(i), reverse=True)
        return items[final_indices]

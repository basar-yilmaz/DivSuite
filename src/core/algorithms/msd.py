import numpy as np
import heapq

from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class MaxSumDiversifier(BaseDiversifier):
    """
    Implement the Max-Sum Dispersion algorithm (Algorithm 6):
      1) While |R| < floor(k/2)*2:
         - find the pair (s_i, s_j) in S that maximizes:
             msd(s_i, s_j) = (1 - lambda_)*(sim(q, s_i) + sim(q, s_j))
                             + 2*lambda_*(1 - cosine_sim(s_i, s_j))
         - move that pair from S to R
      2) If k is odd, choose an arbitrary item s_s from S and add it to R
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
        items: np.ndarray,
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        n = items.shape[0]
        if n == 0:
            return items
        top_k = min(top_k, n)

        if top_k == 0:
            return np.array([])

        titles = items[:, 1].tolist()
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        def relevance(i):
            return float(items[i, 2])

        def divergence(i, j):
            # Ensure indices are valid if sim_matrix is sparse/dict
            return 1.0 - sim_matrix[i, j]

        def msd_score(i, j):
            return (1 - self.lambda_) * (
                relevance(i) + relevance(j)
            ) + 2 * self.lambda_ * divergence(i, j)

        # --- Precompute and Heap ---
        pairwise_scores_heap = []  # Use as a min-heap with negative scores

        # 1. Precompute all scores (O(N^2))
        for i in range(n):
            for j in range(i + 1, n):
                score = msd_score(i, j)
                # Push negative score for max-heap behavior
                heapq.heappush(pairwise_scores_heap, (-score, i, j))
                # Heap build takes O(N^2 log N) overall

        R = []  # Result list of indices
        available_items = set(range(n))  # O(1) average lookup/delete
        num_pairs = top_k // 2

        # 2. Select k//2 pairs using the heap (O(N^2 log N) total pops worst case)
        pairs_added = 0
        while pairs_added < num_pairs and len(available_items) >= 2:
            best_pair_found = False
            while pairwise_scores_heap:  # Find highest-scoring *valid* pair
                _, i, j = heapq.heappop(pairwise_scores_heap)

                if i in available_items and j in available_items:
                    # Found the best valid pair for this iteration
                    R.extend([i, j])
                    available_items.remove(i)
                    available_items.remove(j)
                    pairs_added += 1
                    best_pair_found = True
                    break  # Move to the next pair selection
                # Else: Pair is invalid (one or both items already taken), continue popping

            if not best_pair_found and not pairwise_scores_heap:
                # Heap became empty before finding enough pairs
                break

        # 3. Handle odd k (O(N) or better if items sorted by relevance)
        if top_k % 2 == 1 and len(available_items) > 0:
            # Find highest relevance item among remaining available items
            best_extra = -1
            max_rel = -float("inf")
            for item_idx in available_items:
                rel = relevance(item_idx)
                if rel > max_rel:
                    max_rel = rel
                    best_extra = item_idx
            if best_extra != -1:
                R.append(best_extra)
                # No need to remove from available_items now

        return items[R]

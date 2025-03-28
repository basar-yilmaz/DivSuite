import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
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
        similarity_scores_path: str = "/mnt/scratch1/byilmaz/data_syn/similarity_results.pkl",
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

        # All candidate indices
        all_indices = set(range(num_items))

        # --- PHASE 1: greedily pick items by relevance until R has size k
        sorted_by_rel = sorted(all_indices, key=lambda i: items[i, 2], reverse=True)

        R = []
        S = set(sorted_by_rel)  # unselected candidates

        while len(R) < top_k and len(S) > 0:
            best_s = max(S, key=lambda idx: items[idx, 2])  # pick the highest relevance
            S.remove(best_s)
            R.append(best_s)

        # PHASE 2: swapping
        R_set = set(R)

        while len(S) > 0:
            # pick next item s_s from S with the highest relevance
            s_s = max(S, key=lambda idx: items[idx, 2])
            S.remove(s_s)

            R_prime = set(R_set)  # copy
            best_score = score_set(R_prime)

            # try swapping each s_j in R with s_s
            for s_j in R_set:
                temp = set(R_set)
                temp.remove(s_j)
                temp.add(s_s)
                temp_score = score_set(temp)
                if temp_score > best_score:
                    best_score = temp_score
                    R_prime = temp

            # if improvement, accept it
            old_score = score_set(R_set)
            if best_score > old_score:
                R_set = R_prime

        # Return the final set in descending relevance order
        final_indices = list(R_set)
        return items[final_indices]

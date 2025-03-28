import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
from src.core.embedders.base_embedder import BaseEmbedder
from src.metrics.metrics_utils import load_similarity_scores


class MMRDiversifier(BaseDiversifier):
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
        self, items: np.ndarray, title2embedding: dict = None, top_k: int = 10, **kwargs
    ) -> np.ndarray:
        if items.shape[0] == 0:
            return items

        if not 0 <= self.lambda_ <= 1:
            raise ValueError("lambda_ must be between 0 and 1")

        # if top_k is greater than the number of items, set it to the number of items
        top_k = min(top_k, items.shape[0])

        # Sort items by relevance
        sorted_idx = np.argsort(items[:, 2].astype(float))[::-1]
        items = items[sorted_idx]

        # Calculate similarity matrix using the base class method
        titles = items[:, 1].tolist()
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        selected_indices = [0]  # Pick the most relevant item

        while len(selected_indices) < top_k:
            best_candidate = None
            best_score = -np.inf

            for i in range(len(items)):
                if i in selected_indices:
                    continue

                # Relevance term
                relevance = (1 - self.lambda_) * float(items[i, 2])

                # Diversity term
                diversity_sum = sum((1 - sim_matrix[i, j]) for j in selected_indices)
                diversity_term = (self.lambda_ / len(selected_indices)) * diversity_sum

                # MMR score as per the formula
                mmr_score = relevance + diversity_term

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = i

            if best_candidate is None:
                break

            selected_indices.append(best_candidate)

        return items[selected_indices]

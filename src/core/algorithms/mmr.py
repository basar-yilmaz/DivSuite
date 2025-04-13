import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.core.embedders.base_embedder import BaseEmbedder


class MMRDiversifier(BaseDiversifier):
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

        # --- MMR Selection Loop ---
        while len(selected_indices) < top_k:
            unselected_mask = np.ones(len(items), dtype=bool)
            unselected_mask[selected_indices] = False
            unselected = np.where(unselected_mask)[0]

            # If no candidates remain, exit the loop
            if unselected.size == 0:
                break

            # Vectorize relevance computation for unselected items:
            relevance_candidates = (1 - self.lambda_) * items[unselected, 2].astype(
                float
            )

            # Vectorize diversity: sum (1 - similarity) for each candidate over currently selected items
            diversity_candidates = (self.lambda_ / len(selected_indices)) * np.sum(
                1 - sim_matrix[unselected][:, selected_indices], axis=1
            )

            # Compute the final MMR scores for the unselected candidates
            mmr_scores = relevance_candidates + diversity_candidates

            # Select the candidate with the highest score
            best_candidate = unselected[np.argmax(mmr_scores)]
            selected_indices.append(best_candidate)

        return items[selected_indices]

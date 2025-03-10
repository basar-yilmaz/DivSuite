import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
from src.core.embedders.base_embedder import BaseEmbedder


class MMRDiversifier(BaseDiversifier):
    def __init__(
        self,
        embedder: BaseEmbedder,
        lambda_: float = 0.5,
    ):
        self.lambda_ = lambda_
        self.embedder = embedder

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

        # Extract titles to encode
        titles = items[:, 1].tolist()
        if title2embedding is not None:
            # Use precomputed embeddings.
            try:
                embeddings = np.stack([title2embedding[title] for title in titles])
            except KeyError as e:
                raise ValueError(f"Missing embedding for title: {e}")
        else:
            # Fall back to computing embeddings on the fly.
            embeddings = self.embedder.encode_batch(titles)
        sim_matrix = compute_pairwise_cosine(embeddings)  # N x N

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

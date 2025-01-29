import numpy as np
from algorithms.base import BaseDiversifier
from utils import compute_pairwise_cosine
from embedders.ste_embedder import STEmbedder
from embedders.hf_embedder import HFEmbedder
from config import DEFAULT_EMBEDDER

class MMRDiversifier(BaseDiversifier):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.device = device
        if DEFAULT_EMBEDDER == STEmbedder:
            self.embedder = STEmbedder(
                model_name=model_name, device=device, batch_size=batch_size
            )
        else:
            self.embedder = HFEmbedder(
                model_name=model_name, device=device, max_chunk_size=batch_size
            )

    def diversify(
        self, items: np.ndarray, top_k: int = 10, lambda_: float = 0.5, **kwargs
    ) -> np.ndarray:
        if items.shape[0] == 0:
            return items

        if not 0 <= lambda_ <= 1:
            raise ValueError("lambda_ must be between 0 and 1")

        # if top_k is greater than the number of items, set it to the number of items
        top_k = min(top_k, items.shape[0])

        # Sort items by relevance
        sorted_idx = np.argsort(items[:, 2].astype(float))[::-1]
        items = items[sorted_idx]

        # Early return no diversification needed
        if top_k == 1 or lambda_ == 1.0:
            return items[:top_k]

        # Extract titles to encode
        titles = items[:, 1].tolist()
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
                relevance = (1 - lambda_) * float(items[i, 2])

                # Diversity term
                diversity_sum = sum((1 - sim_matrix[i, j]) for j in selected_indices)
                diversity_term = (lambda_ / len(selected_indices)) * diversity_sum

                # MMR score as per the formula
                mmr_score = relevance + diversity_term

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = i

            if best_candidate is None:
                break

            selected_indices.append(best_candidate)

        return items[selected_indices]

    @property
    def embedder_type(self) -> str:
        return "STEmbedder" if isinstance(self.embedder, STEmbedder) else "HFEmbedder"

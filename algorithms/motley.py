import numpy as np
from algorithms.base import BaseDiversifier
from utils import compute_pairwise_cosine
from embedders.ste_embedder import STEmbedder
from embedders.hf_embedder import HFEmbedder
from config import DEFAULT_EMBEDDER


class MotleyDiversifier(BaseDiversifier):
    """
    Implements the Motley algorithm (Algorithm 5), which:
      1) Picks the highest-relevance item s_s from S, moves it to R
      2) While |R| < k:
           - pick s_s from S with highest relevance
           - if for all s_r in R, div(s_r, s_s) >= threshold => R <- R + { s_s }
    """

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
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance]
        top_k: int = 10,
        div_threshold: float = 0.5,  # theta' in the pseudo-code
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array: [item_id, title, relevance_score]
        :param top_k: number of items we want in R
        :param div_threshold: the threshold θ' for delta_div(s_r, s_s)
                              (e.g., 1 - cosine_similarity >= div_threshold)
        :return: A subset of items, up to size k, that meets the Motley criterion.
        """

        num_items = items.shape[0]
        if num_items == 0:
            return items

        # If fewer items than k, just clamp top_k
        top_k = min(top_k, num_items)

        # 1) Embed items to compute pairwise similarity => from that, we get distances
        titles = items[:, 1].tolist()
        embeddings = self.embedder.encode_batch(titles)  # (N, dim)
        sim_matrix = compute_pairwise_cosine(embeddings)  # NxN

        # distance(i,j) = 1 - sim_matrix[i,j]
        def distance(i, j):
            return 1.0 - sim_matrix[i, j]

        # We'll treat items[i,2] as "relevance"
        def relevance(i):
            return float(items[i, 2])

        # Convert all item indices into a set
        S = set(range(num_items))
        R = []

        # 2) Pick the highest relevance item from S => s_s => R
        s_s = max(S, key=lambda i: relevance(i))
        S.remove(s_s)
        R.append(s_s)

        # 3) While |R| < k:
        while len(R) < top_k and len(S) > 0:
            # pick next highest relevance item
            next_s = max(S, key=lambda i: relevance(i))
            S.remove(next_s)

            # check if it's sufficiently diverse from every item in R
            all_diverse = True
            for r_idx in R:
                if distance(r_idx, next_s) < div_threshold:
                    all_diverse = False
                    break

            if all_diverse:
                R.append(next_s)

        # Sort final indices in descending order of relevance or keep as is
        R.sort(key=lambda i: relevance(i), reverse=True)

        return items[R]

    @property
    def embedder_type(self) -> str:
        return "STEmbedder" if isinstance(self.embedder, STEmbedder) else "HFEmbedder"

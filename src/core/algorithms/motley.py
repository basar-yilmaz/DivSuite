import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
from src.core.embedders.base_embedder import BaseEmbedder


class MotleyDiversifier(BaseDiversifier):
    """
    Implements the Motley algorithm (Algorithm 5), which:
      1) Picks the highest-relevance item s_s from S, moves it to R
      2) While |R| < k:
           - pick s_s from S with the highest relevance
           - if for all s_r in R, div(s_r, s_s) >= threshold => R <- R + { s_s }
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        theta_: float = 0.5,
    ):
        self.theta_ = theta_
        self.embedder = embedder

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance]
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array: [item_id, title, relevance_score]
        :param top_k: number of items we want in R
        :return: A subset of items, up to size k, that meets the Motley criterion.
        """

        num_items = items.shape[0]
        if num_items == 0:
            return items

        # If fewer items than k, just clamp top_k
        top_k = min(top_k, num_items)

        # 1) Embed items to compute pairwise similarity => from that, we get distances
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
                if distance(r_idx, next_s) < self.theta_:
                    all_diverse = False
                    break

            if all_diverse:
                R.append(next_s)

        return items[R]

import numpy as np
from algorithms.base import BaseDiversifier
from utils import compute_pairwise_cosine
from embedders.base_embedder import BaseEmbedder


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
    ):
        self.lambda_ = lambda_
        self.embedder = embedder

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3) => [id, title, relevance_score]
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array of [item_id, title, relevance]
                      where items[i,2] = sim(q, s_i) is the 'relevance' for s_i.
        :param top_k: result set size k
        :param lambda_: parameter in [0,1], balancing relevance vs. diversity
        :return: A subset of items (size k or fewer if N < k).
        """

        n = items.shape[0]
        if n == 0:
            return items
        top_k = min(top_k, n)

        # Embed items & compute NxN similarity
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
        sim_matrix = compute_pairwise_cosine(embeddings)  # NxN in [0,1]

        def relevance(i):
            return float(items[i, 2])

        def divergence(i, j):
            return 1.0 - sim_matrix[i, j]

        def msd_score(i, j):
            return (1 - self.lambda_) * (
                relevance(i) + relevance(j)
            ) + 2 * self.lambda_ * divergence(i, j)

        # Convert to a set or list for selecting pairs
        S = set(range(n))  # candidate indices
        R = []  # result set (list of indices)

        # We'll add pairs until we have 2*(k//2) items in R (the largest even number ≤ k)
        num_pairs = top_k // 2

        # While we have pairs to pick and enough items in S
        for _ in range(num_pairs):
            if len(S) < 2:
                break  # not enough items left for a pair

            # Find the pair (i,j) in S that maximizes msd_score(i,j)
            best_score = float("-inf")
            best_pair = None
            s_list = list(S)

            # Naive O(|S|^2) search
            for idx1 in range(len(s_list)):
                i = s_list[idx1]
                for idx2 in range(idx1 + 1, len(s_list)):
                    j = s_list[idx2]
                    score_ij = msd_score(i, j)
                    if score_ij > best_score:
                        best_score = score_ij
                        best_pair = (i, j)

            if best_pair is None:
                break

            # Add that pair to R, remove from S
            i, j = best_pair
            R.extend([i, j])
            S.remove(i)
            S.remove(j)

        # If k is odd, we add one more item from S
        if top_k % 2 == 1:
            if len(S) > 0:
                # The algorithm says "choose an arbitrary object s_i ∈ S"
                # We'll pick the highest relevance item or a random item.
                # For "arbitrary," let's just pick the highest relevance for consistency:
                extra = max(S, key=lambda idx: relevance(idx))
                R.append(extra)
                S.remove(extra)

        return items[R]

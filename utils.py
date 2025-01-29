import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """
    Given a 2D array of shape (N, dim) of embeddings,
    return an NxN matrix of pairwise cosine similarities.
    """

    # TODO this can be optimized by reducing space complexity
    return cosine_similarity(embeddings, embeddings)

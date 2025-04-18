"""
Base Diversifier Class

This is the base class for all diversification algorithms.
In order to implement a new diversifier, you need extend this class and implement the `diversify` method.
"""

from abc import ABC, abstractmethod
import numpy as np
from src.metrics.metrics_utils import load_similarity_scores
from sklearn.metrics.pairwise import cosine_similarity


class BaseDiversifier(ABC):
    """
    Abstract base class for all diversification algorithms.
    """

    def __init__(
        self,
        embedder=None,
        use_similarity_scores: bool = False,
        item_id_mapping: dict = None,
        similarity_scores_path: str = None,
        **kwargs,
    ):
        """
        Initialize the base diversifier with common parameters.

        :param embedder: The embedder to use for generating embeddings
        :param use_similarity_scores: Whether to use pre-computed similarity scores
        :param item_id_mapping: Mapping from item titles/text to IDs for similarity lookup
        :param similarity_scores_path: Path to the similarity scores file
        :param kwargs: Additional parameters specific to child classes
        """
        self.embedder = embedder
        self.use_similarity_scores = use_similarity_scores
        self.item_id_mapping = item_id_mapping
        self.similarity_scores_path = similarity_scores_path

        # Pre-load similarity scores if using them
        if self.use_similarity_scores:
            self.sim_scores_dict = load_similarity_scores(self.similarity_scores_path)

    def compute_similarity_matrix(
        self, titles, title2embedding=None, precomputed_sim_matrix=None
    ):
        """
        Compute similarity matrix using either pre-loaded similarity scores or embeddings.

        :param titles: List of item titles/text
        :param title2embedding: Optional pre-computed embeddings dict
        :param precomputed_sim_matrix: Optional pre-computed similarity matrix
        :return: NxN similarity matrix
        """
        # If a precomputed similarity matrix is provided, use it
        if precomputed_sim_matrix is not None:
            return precomputed_sim_matrix

        if self.use_similarity_scores:
            # Use preloaded similarity scores to build similarity matrix
            sim_matrix = np.zeros((len(titles), len(titles)))

            # Get item IDs for the titles
            item_ids = [self.item_id_mapping.get(title) for title in titles]

            # Fill the similarity matrix using cached similarity scores
            for i in range(len(item_ids)):
                for j in range(len(item_ids)):
                    if i == j:
                        sim_matrix[i, j] = 1.0  # Self-similarity is 1
                    else:
                        id1, id2 = item_ids[i], item_ids[j]
                        if id1 is None or id2 is None:
                            sim_matrix[i, j] = 0.0
                            continue

                        # Look up similarity score in both directions
                        sim = self.sim_scores_dict.get((id1, id2))
                        if sim is None:
                            sim = self.sim_scores_dict.get((id2, id1))

                        sim_matrix[i, j] = sim if sim is not None else 0.0
        else:
            # Use embeddings-based similarity
            if title2embedding is not None:
                # Use precomputed embeddings
                try:
                    embeddings = np.stack([title2embedding[title] for title in titles])
                except KeyError as e:
                    raise ValueError(f"Missing embedding for title: {e}")
            else:
                # Compute embeddings on the fly
                if self.embedder is None:
                    raise ValueError(
                        "Embedder must be provided when title2embedding is None"
                    )
                embeddings = self.embedder.encode_batch(titles)

            sim_matrix = cosine_similarity(embeddings)

        return sim_matrix

    @abstractmethod
    def diversify(self, items: np.ndarray, top_k: int = 10, **kwargs) -> np.ndarray:
        """
        Diversify the given array of items (shape: N x 3, e.g. [id, title, sim_score])
        and return a smaller or reordered array.

        :param items: A 2D array of shape (N, 3), for example:
                      [
                        [movie_id, movie_title, sim_score],
                        ...
                      ]
        :param top_k: How many items to retrieve after diversification.
        :param kwargs: Additional algorithm-specific parameters (e.g. lambda).
                       May include:
                       - title2embedding: Dict mapping titles to precomputed embeddings
                       - precomputed_sim_matrix: Precomputed similarity matrix for these items
        :return: A 2D array of shape (top_k, 3), diversified.
        """
        pass

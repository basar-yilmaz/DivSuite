import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
from src.core.embedders.base_embedder import BaseEmbedder
from sklearn_extra.cluster import KMedoids


class CLTDiversifier(BaseDiversifier):
    """
    Implements a cluster-based diversification approach (Algorithm 7: CLT).
    1) Embed items and compute distance = 1 - similarity for each pair.
    2) Run k-medoids with distance matrix => clusters C = {c1, c2, ..., ck}.
    3) For each cluster c_i, pick one representative item s_i (e.g. the medoid or highest relevance).
    4) R <- R ∪ {s_i}.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
    ):
        """
        :param model_name: Hugging Face or SentenceTransformers model name.
        :param device: 'cpu' or 'cuda'
        :param batch_size: batch size for embedding (depending on embedder).
        """
        self.embedder = embedder

    def diversify(
        self,
        items: np.ndarray,  # shape (N, 3) => [id, title, relevance_score]
        top_k: int = 10,
        pick_strategy: str = "medoid",  # or "highest_relevance"
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array of [id, title, relevance_score].
        :param top_k: number of clusters => also how many items we'll pick.
        :param pick_strategy:
           - 'medoid': pick the medoid for each cluster
           - 'highest_relevance': pick the item with the highest items[i,2]
        :return: A subset of items of size k (or fewer if items < k).
        """
        num_items = items.shape[0]
        if num_items == 0:
            return items

        if top_k <= 2:
            raise ValueError("CLT requires at least 3 clusters (top_k > 2).")

        top_k = min(top_k, num_items)

        # 1) Embed items
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

        # 2) Build a distance matrix = 1 - cosine_similarity
        sim_matrix = compute_pairwise_cosine(embeddings)
        distance_matrix = 1.0 - sim_matrix

        # set 1 on diagonal to avoid floating point errors
        np.fill_diagonal(distance_matrix, 0)

        # 3) KMedoids
        kmed = KMedoids(
            n_clusters=top_k,
            metric="precomputed",  # means we pass a precomputed distance matrix
            method="pam",
            init="k-medoids++",
            max_iter=300,
            random_state=42,
        )
        kmed.fit(distance_matrix)

        # cluster labels, medoid indices
        labels = kmed.labels_  # shape (N,)
        medoid_indices = kmed.medoid_indices_  # length k

        # 4) For each cluster, pick one item
        # Option A: pick the medoid (we use this one for now #TODO Sengör hocaya sor)
        # Option B: pick the item with highest relevance in that cluster
        chosen_indices = []

        for cluster_id in range(top_k):
            # All items that belong to cluster_id
            cluster_item_indices = np.where(labels == cluster_id)[0]

            if pick_strategy == "medoid":
                # The medoid index for cluster_id is medoid_indices[cluster_id]
                chosen_idx = medoid_indices[cluster_id]
            else:
                # 'highest_relevance': pick item in cluster with max items[i,2]
                # cluster_item_indices might have multiple items
                chosen_idx = max(cluster_item_indices, key=lambda i: items[i, 2])

            chosen_indices.append(chosen_idx)

        # 5) Return those chosen items
        # Optionally, reorder them by descending relevance, or keep cluster order
        chosen_indices = list(set(chosen_indices))  # ensure unique (should be already)

        return items[chosen_indices]

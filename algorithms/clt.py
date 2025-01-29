import numpy as np
from algorithms.base import BaseDiversifier
from utils import compute_pairwise_cosine
from embedders.ste_embedder import STEmbedder
from embedders.hf_embedder import HFEmbedder
from sklearn_extra.cluster import KMedoids

DEFAULT_EMBEDDER = STEmbedder


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
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        :param model_name: Hugging Face or SentenceTransformers model name.
        :param device: 'cpu' or 'cuda'
        :param batch_size: batch size for embedding (depending on embedder).
        """
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
        items: np.ndarray,  # shape (N, 3) => [id, title, relevance_score]
        top_k: int = 10,
        pick_strategy: str = "medoid",  # or "highest_relevance"
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
        embeddings = self.embedder.encode_batch(titles)  # shape (N, embed_dim)

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
        chosen_indices.sort(key=lambda i: items[i, 2], reverse=True)

        return items[chosen_indices]

    @property
    def embedder_type(self) -> str:
        return "STEmbedder" if isinstance(self.embedder, STEmbedder) else "HFEmbedder"

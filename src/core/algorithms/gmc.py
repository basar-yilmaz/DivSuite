import numpy as np
from src.core.algorithms.base import BaseDiversifier
from src.utils.utils import compute_pairwise_cosine
from src.core.embedders.base_embedder import BaseEmbedder


class GMCDiversifier(BaseDiversifier):
    """
    Implements the Greedy Marginal Contribution (GMC) diversification algorithm.
    """

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
        self,
        items: np.ndarray,  # shape (N, 3): [id, title, relevance]
        top_k: int = 10,
        title2embedding: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array, e.g. [item_id, title, relevance_score]
        :param top_k: desired size of the final diversified set R
        :param lambda_: trade-off parameter between relevance and diversity
        :return: A subset of 'items' of size top_k (or fewer if items < top_k)
        """

        num_items = items.shape[0]
        if num_items == 0:
            return items

        # Clamp top_k if necessary
        top_k = min(top_k, num_items)

        # Calculate similarity matrix using the base class method
        titles = items[:, 1].tolist()  # item text
        sim_matrix = self.compute_similarity_matrix(titles, title2embedding)

        # Initialize empty result set
        R = []
        S = set(range(num_items))  # Candidate pool

        while len(R) < top_k and len(S) > 0:
            best_s = max(S, key=lambda s: self.mmc(s, R, S, items, sim_matrix, top_k))
            S.remove(best_s)
            R.append(best_s)

        return items[R]

    def mmc(self, s_i, R, S, items, sim_matrix, k):
        """
        Computes the Maximum Marginal Contribution (MMC) score for an item.
        """
        if not R:
            return (1 - self.lambda_) * items[s_i, 2]  # İlk eleman için sadece alaka skoru kullanılır.

        p = len(R) + 1  # Yeni eleman eklendiğinde R'nin boyutu
        k_minus_1 = max(1, k - 1)  # (k-1) normalizasyon için

        # 1. Alaka katkısı
        sim_term = (1 - self.lambda_) * items[s_i, 2]

        # 2. Çeşitlilik katkısı (R içinde)
        div_sum_R = sum(1 - sim_matrix[s_i, s_j] for s_j in R)
        div_term_R = (self.lambda_ / k_minus_1) * div_sum_R

        # 3. Çeşitlilik katkısı (S içinde, en büyük k - p skorlarını alma)
        remaining_depth = k - p
        div_term_S = 0

        if remaining_depth > 0:
            # S'deki tüm çeşitlilik katkılarını hesapla ve sırala
            diversity_scores = [(1 - sim_matrix[s_i, s_j]) for s_j in S if s_j != s_i]
            top_l_scores = sorted(diversity_scores, reverse=True)[:remaining_depth]  # En büyük k - p kadarını seç

            # Seçilen en büyük katkıları topla
            div_term_S = (self.lambda_ / k_minus_1) * sum(top_l_scores)

        return sim_term + div_term_R + div_term_S




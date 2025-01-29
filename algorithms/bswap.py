import numpy as np
from algorithms.base import BaseDiversifier
from utils import compute_pairwise_cosine
from embedders.ste_embedder import STEmbedder
from embedders.hf_embedder import HFEmbedder
from config import DEFAULT_EMBEDDER


class BSwapDiversifier(BaseDiversifier):
    """
    Implements the BSWAP algorithm (Algorithm 3).

    Steps (roughly):
      1) R <- empty
      2) While |R| < k: pick s_s in S with highest relevance, move from S to R
      3) s_d <- argmin_{si in R} (divContribution(R, si))
      4) s_s <- argmax_{si in S} (relevance)
      5) while [deltaSim(q, s_d) - deltaSim(q, s_s) <= theta] and |S|>0:
          if div( (R \ {s_d}) U {s_s} ) > div(R):
             R <- (R \ {s_d}) U {s_s}
             s_d <- argmin_{si in R} (divContribution(R, si))
          s_s <- argmax_{si in S} (relevance)
          S <- S \ {s_s}
    """

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32):
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
        items: np.ndarray,  # shape: (N, 3) => [id, title, relevance_score]
        top_k: int = 10,
        theta: float = 0.1,
        **kwargs,
    ) -> np.ndarray:
        """
        :param items: Nx3 array with columns [id, title, relevance_score].
        :param top_k: result set size k (|R|=k).
        :param theta: distance threshold in the condition
                      deltaSim(q, s_d) - deltaSim(q, s_s) <= theta
        :return: A subset of items of size k (or fewer if N < k).
        """

        num_items = items.shape[0]
        if num_items == 0:
            return items
        top_k = min(top_k, num_items)

        # 1) Embed items & build similarity matrix
        titles = items[:, 1].tolist()
        embeddings = self.embedder.encode_batch(titles)
        sim_matrix = compute_pairwise_cosine(embeddings)  # NxN in [0,1]

        # We treat items[i,2] as the "relevance" = δsim(q, s_i)
        def relevance(i: int) -> float:
            return float(items[i, 2])

        # sum of pairwise (1 - sim) for the entire set R
        def diversity_of_set(R_set) -> float:
            R_list = list(R_set)
            div_val = 0.0
            for i in range(len(R_list)):
                for j in range(i + 1, len(R_list)):
                    div_val += 1.0 - sim_matrix[R_list[i], R_list[j]]
            return div_val

        # diversity contribution of a single item s_i in R
        # sum(1 - sim(i,j)) for j in R, j != i
        def div_contribution(R_set, i: int) -> float:
            contrib = 0.0
            for j in R_set:
                if j != i:
                    contrib += 1.0 - sim_matrix[i, j]
            return contrib

        # all candidate indices
        S = set(range(num_items))
        R = set()

        # 2) While |R| < k, pick s_s from S with highest relevance, move it to R
        while len(R) < top_k and len(S) > 0:
            s_s = max(S, key=lambda idx: relevance(idx))  # highest rel
            S.remove(s_s)
            R.add(s_s)

        # if we already have k items or S is empty, we skip the BSWAP loop
        if len(R) < 1 or len(S) < 1:
            # Return in descending relevance order
            final_indices = list(R)
            final_indices.sort(key=lambda i: relevance(i), reverse=True)
            return items[final_indices]

        # 3) s_d = argmin_{ si in R } divContribution(R, si)
        def pick_s_d(R_set):
            return min(R_set, key=lambda x: div_contribution(R_set, x))

        s_d = pick_s_d(R)

        # 4) s_s = argmax_{ si in S } relevance
        def pick_s_s(S_set):
            return max(S_set, key=lambda x: relevance(x))

        s_s = pick_s_s(S)
        S.remove(s_s)

        # 5) while deltaSim(q, s_d) - deltaSim(q, s_s) <= theta and |S|>0 do ...
        while (relevance(s_d) - relevance(s_s) <= theta) and len(S) >= 0:
            # if div( (R \ s_d) U { s_s } ) > div(R) then R = ...
            old_div = diversity_of_set(R)
            R_candidate = set(R)
            R_candidate.remove(s_d)
            R_candidate.add(s_s)

            new_div = diversity_of_set(R_candidate)
            if new_div > old_div:
                R = R_candidate
                # update s_d
                if len(R) > 0:
                    s_d = pick_s_d(R)
                else:
                    break

            # pick a new s_s from S if any left
            if len(S) == 0:
                break
            s_s = pick_s_s(S)
            S.remove(s_s)

            # re-check the condition
            if relevance(s_d) - relevance(s_s) > theta:
                # condition fails => break
                break

        # Done. Return R in descending relevance order
        final_indices = list(R)
        return items[final_indices]

    @property
    def embedder_type(self) -> str:
        if isinstance(self.embedder, STEmbedder):
            return "STEmbedder"
        else:
            return "HFEmbedder"

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Given a list of strings, return a 2D NumPy array (N, embedding_dim).
        """
        pass

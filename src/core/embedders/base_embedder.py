"""
Base Embedder Class

This is the base class for all embedders. It defines the interface for all embedders.
In order to implement a new embeddder, you need extend this class and implement the `encode_batch` method.
"""

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

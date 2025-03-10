"""
Base Diversifier Class

This is the base class for all diversification algorithms.
In order to implement a new diversifier, you need extend this class and implement the `diversify` method.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDiversifier(ABC):
    """
    Abstract base class for all diversification algorithms.
    """

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
        :return: A 2D array of shape (top_k, 3), diversified.
        """
        pass

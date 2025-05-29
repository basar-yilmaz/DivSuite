"""
Sentence Transformers Embedder Class
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.embedders.base_embedder import BaseEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__, level="INFO")


class STEmbedder(BaseEmbedder):
    """
    A simple embedder that uses SentenceTransformers' built-in `.encode()`
    to handle batching, GPU usage, etc.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 1024,
    ):
        """
        :param model_name: A model hosted on Hugging Face, e.g. "all-MiniLM-L6-v2"
        :param device: Device to run on, e.g. 'cpu' or 'cuda'
        :param batch_size: Batch size for encoding
        """
        # logger.info(f"Initializing STEmbedder with model={model_name} on {device}")
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )
        self.batch_size = batch_size
        self.pool = None
        self.pool = self.model.start_multi_process_pool()

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of strings into a 2D NumPy array of shape (N, embed_dim).
        """
        logger.debug(
            f"Encoding batch of {len(texts)} texts with batch_size={self.batch_size}"
        )
        embeddings = self.model.encode_multi_process(
            texts, self.pool, batch_size=self.batch_size, show_progress_bar=True
        )
        logger.debug(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def __del__(self):
        # Clean up pool when the object is destroyed
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)

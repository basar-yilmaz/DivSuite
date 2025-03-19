"""
Gemini Embedder Class

This class uses the Gemini API to create embeddings for text.
"""

from google import genai
import numpy as np
from typing import List

from src.core.embedders.base_embedder import BaseEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__, level="INFO")


class GeminiEmbedder(BaseEmbedder):
    """
    An embedder that uses Google's Gemini API to create embeddings.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-exp-03-07",
        batch_size: int = 32,
    ):
        """
        Initialize the Gemini embedder.

        :param api_key: Gemini API key
        :param model_name: Gemini embedding model name
        :param batch_size: Batch size for encoding
        """
        logger.info(f"Initializing GeminiEmbedder with model={model_name}")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.batch_size = batch_size

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of strings into a 2D NumPy array of shape (N, embed_dim).
        """
        logger.debug(f"Encoding batch of {len(texts)} texts with Gemini API")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = []

            # Process each text in the batch
            for text in batch:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                batch_embeddings.append(result.embeddings)

            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        logger.debug(f"Generated embeddings with shape {embeddings_array.shape}")

        return embeddings_array

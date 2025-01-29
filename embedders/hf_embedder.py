# embedders/hf_embedder.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List

from embedders.base_embedder import BaseEmbedder
from logger import get_logger

logger = get_logger(__name__, level="INFO")


class HFEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        max_chunk_size: int = 32,
    ):
        """
        :param model_name: Hugging Face model name
        :param device: 'cpu' or 'cuda'
        :param max_chunk_size: how many texts to process at once (batch size)
        """
        logger.info(f"Initializing HFEmbedder with model={model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # If multiple GPUs are available, we can wrap the model in DataParallel
        # Alternatively, we do manual chunk distribution below.
        if torch.cuda.device_count() > 1 and device == "cuda":
            logger.info(f"{torch.cuda.device_count()} GPUs found. Using DataParallel.")
            self.model = torch.nn.DataParallel(self.model)

        self.device = device
        self.model.to(self.device)
        self.max_chunk_size = max_chunk_size

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode the list of texts in chunks. Returns a 2D NumPy array (N, dim).
        """
        all_embeddings = []
        start_idx = 0

        while start_idx < len(texts):
            end_idx = min(start_idx + self.max_chunk_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # mean pooling here
            last_hidden = (
                outputs.last_hidden_state
            )  # shape: [batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"]
            # Expand attention mask to match hidden size
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            masked_hidden = last_hidden * mask_expanded
            summed = torch.sum(masked_hidden, dim=1)
            counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = summed / counts

            # Normalize (l2)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)

            start_idx = end_idx

        # Concatenate all chunk embeddings
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings

from embedders.ste_embedder import STEmbedder
from embedders.hf_embedder import HFEmbedder  # noqa: F401

# Default embedder to use across the application
DEFAULT_EMBEDDER = STEmbedder

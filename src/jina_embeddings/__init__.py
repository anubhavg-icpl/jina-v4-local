"""
Jina Embeddings v4 - Professional Python Package

A state-of-the-art multimodal embedding library supporting text and image embeddings
with task-specific adapters and Matryoshka representation learning.
"""

__version__ = "1.0.0"
__author__ = "Jina AI"

from jina_embeddings.core.embeddings import JinaEmbeddings
from jina_embeddings.core.model import EmbeddingModel
from jina_embeddings.config.settings import Config

__all__ = [
    "JinaEmbeddings",
    "EmbeddingModel",
    "Config",
    "__version__",
]
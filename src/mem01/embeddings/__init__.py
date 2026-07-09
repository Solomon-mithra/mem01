"""Embedding providers. Read and write paths depend only on Embedder."""

from mem01.embeddings.base import Embedder
from mem01.embeddings.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder"]

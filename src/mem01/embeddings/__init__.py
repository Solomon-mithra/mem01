"""Embedding providers. Read and write paths depend only on Embedder."""

from mem01.embeddings.base import Embedder
from mem01.embeddings.fake import FakeEmbedder
from mem01.embeddings.openai_embedder import OpenAIEmbedder

__all__ = ["Embedder", "FakeEmbedder", "OpenAIEmbedder"]

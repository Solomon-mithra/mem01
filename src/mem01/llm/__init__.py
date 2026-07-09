"""LLM adapters for the write path only (extract_ops).

Read path must never import this package for default recall.
Any provider works if it implements LLMClient — OpenAI-shape, Claude, etc.
"""

from mem01.llm.base import ChatMessage, LLMClient
from mem01.llm.fake import FakeLLM

__all__ = ["ChatMessage", "FakeLLM", "LLMClient"]

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CHAT_MODEL = "deepseek-v4-pro:cloud"
DEFAULT_EMBED_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_BASE_URL = "http://192.168.1.18:11434"
DEFAULT_COLLECTION = "rag_demo_t1"


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str
    embed_model: str
    base_url: str


def model_config(
    chat_model: str | None = None,
    embed_model: str | None = None,
    base_url: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        chat_model=chat_model or os.getenv("RAG_DEMO_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        embed_model=embed_model or os.getenv("RAG_DEMO_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        base_url=base_url or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_BASE_URL),
    )

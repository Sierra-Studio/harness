"""LLM layer: the model-completion Provider contract, its registry, and tokenization."""

from __future__ import annotations

from .provider import (
    AzureFoundryProvider,
    FakeProvider,
    ModelResult,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    Provider,
    build_provider,
    detect_provider,
    provider_label,
)
from .registry import ProviderRegistry, register_provider
from .tokenizer import count_tokens

__all__ = [
    "Provider",
    "ModelResult",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "AzureFoundryProvider",
    "FakeProvider",
    "build_provider",
    "detect_provider",
    "provider_label",
    "ProviderRegistry",
    "register_provider",
    "count_tokens",
]

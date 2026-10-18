"""Provider clients behind the common ProviderClient interface."""

from __future__ import annotations

from visibility_collector.config import ProviderConfig
from visibility_collector.providers.base import (
    CollectionTask,
    ProviderClient,
    ProviderError,
    ProviderResult,
)

__all__ = ["CollectionTask", "ProviderClient", "ProviderError", "ProviderResult", "build_client"]


def build_client(config: ProviderConfig) -> ProviderClient:
    """Create the real SDK-backed client for a configured provider."""
    if config.name == "openai":
        from visibility_collector.providers.openai_provider import OpenAIProvider

        return OpenAIProvider.from_config(config)
    if config.name == "gemini":
        from visibility_collector.providers.gemini_provider import GeminiProvider

        return GeminiProvider.from_config(config)
    raise ValueError(f"Unsupported provider: {config.name}")

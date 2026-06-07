"""
Provider-specific batch processing implementations.

This module contains provider-specific implementations for OpenAI and Anthropic
batch processing APIs.
"""

from __future__ import annotations

from typing import Any
from .base import BatchProvider


def get_provider(provider_name: str) -> BatchProvider:
    """Factory function to get the appropriate provider instance"""
    if provider_name == "openai":
        try:
            from .openai import OpenAIProvider
            return OpenAIProvider()
        except ImportError as err:
            raise ValueError("OpenAI is not installed") from err
    elif provider_name == "anthropic":
        try:
            from .anthropic import AnthropicProvider
            return AnthropicProvider()
        except ImportError as err:
            raise ValueError("Anthropic is not installed") from err
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def __getattr__(name: str) -> Any:
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider
        return OpenAIProvider
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider
        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BatchProvider", "OpenAIProvider", "AnthropicProvider", "get_provider"]

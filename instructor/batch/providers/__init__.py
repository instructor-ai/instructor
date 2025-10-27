"""
Provider-specific batch processing implementations.

This module contains provider-specific implementations for OpenAI and Anthropic
batch processing APIs.
"""

from .base import BatchProvider
import importlib.util

if importlib.util.find_spec("openai") is not None:
    from .openai import OpenAIProvider
if importlib.util.find_spec("anthropic") is not None:
    from .anthropic import AnthropicProvider


def get_provider(provider_name: str) -> BatchProvider:
    """Factory function to get the appropriate provider instance"""
    if provider_name == "openai":
        if OpenAIProvider is None:
            raise ValueError("OpenAI is not installed")
        return OpenAIProvider()
    elif provider_name == "anthropic":
        if AnthropicProvider is None:
            raise ValueError("Anthropic is not installed")
        return AnthropicProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


__all__ = ["BatchProvider", "OpenAIProvider", "AnthropicProvider", "get_provider"]

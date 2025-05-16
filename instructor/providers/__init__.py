"""
Provider registry and interface for Instructor.

This module defines the provider protocol and registry for Instructor.
Providers should be registered here to be discoverable by the auto_client.
"""

from typing import Optional, TypeVar

# Type for provider base class
T = TypeVar("T", bound="ProviderBase")

# Global registry of providers
_provider_registry: dict[str, type[T]] = {}


def register_provider(name: str):
    """
    Decorator to register a provider class.

    Args:
        name: Unique provider identifier
    """

    def decorator(cls: type[T]) -> type[T]:
        _provider_registry[name] = cls
        return cls

    return decorator


def get_provider(name: str) -> Optional[type[T]]:
    """
    Get provider by name.

    Args:
        name: Provider name

    Returns:
        Provider class or None if not found
    """
    return _provider_registry.get(name)


def list_providers() -> list[str]:
    """
    List all registered providers.

    Returns:
        List of provider names
    """
    return list(_provider_registry.keys())

"""Provider registry for automatic provider discovery and management."""

from typing import Optional, Callable
from contextlib import contextmanager

from ..mode import Mode
from ..exceptions import ProviderNotFoundError, InvalidModeError
from .base import BaseProvider


class ProviderRegistry:
    """Registry for managing instructor providers."""

    _providers: dict[str, type[BaseProvider]] = {}
    _instances: dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseProvider]], type[BaseProvider]]:
        """
        Decorator to register a provider class.

        Usage:
            @ProviderRegistry.register("openai")
            class OpenAIProvider(BaseProvider):
                ...
        """

        def decorator(provider_class: type[BaseProvider]) -> type[BaseProvider]:
            cls._providers[name.lower()] = provider_class
            return provider_class

        return decorator

    @classmethod
    def get_provider(cls, name: str) -> BaseProvider:
        """Get a provider instance by name."""
        name = name.lower()

        # Return cached instance if available
        if name in cls._instances:
            return cls._instances[name]

        # Create new instance
        if name not in cls._providers:
            raise ProviderNotFoundError(
                f"Provider '{name}' not found. Available providers: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[name]
        instance = provider_class()
        cls._instances[name] = instance
        return instance

    @classmethod
    def get_provider_for_mode(cls, mode: Mode) -> Optional[BaseProvider]:
        """Find the first provider that supports the given mode."""
        for name, provider_class in cls._providers.items():
            if name not in cls._instances:
                cls._instances[name] = provider_class()

            provider = cls._instances[name]
            if provider.supports_mode(mode):
                return provider

        return None

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers and instances."""
        cls._providers.clear()
        cls._instances.clear()

    @classmethod
    @contextmanager
    def temporary_provider(cls, name: str, provider_class: type[BaseProvider]):
        """Context manager for temporarily registering a provider."""
        # Store original state
        original_provider = cls._providers.get(name)
        original_instance = cls._instances.get(name)

        # Register temporary provider
        cls._providers[name] = provider_class
        if name in cls._instances:
            del cls._instances[name]

        try:
            yield
        finally:
            # Restore original state
            if original_provider:
                cls._providers[name] = original_provider
                if original_instance:
                    cls._instances[name] = original_instance
            else:
                cls._providers.pop(name, None)
                cls._instances.pop(name, None)

    @classmethod
    def validate_mode_support(cls, provider_name: str, mode: Mode) -> None:
        """Validate that a provider supports a given mode."""
        provider = cls.get_provider(provider_name)
        if not provider.supports_mode(mode):
            raise InvalidModeError(
                f"Provider '{provider_name}' does not support mode '{mode}'. "
                f"Supported modes: {provider.get_supported_modes()}"
            )

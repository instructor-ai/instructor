"""Mode handler registry for v2.

Central registry for all (Provider, ModeType) combinations and their handlers.
Supports lazy loading, dynamic registration, and queryable API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from instructor.v2.core.mode_types import Mode, ModeType, Provider
from instructor.v2.core.protocols import ReaskHandler, RequestHandler, ResponseParser


@dataclass
class ModeHandlers:
    """Collection of handlers for a specific mode."""

    request_handler: RequestHandler
    reask_handler: ReaskHandler
    response_parser: ResponseParser


class ModeRegistry:
    """Central registry for mode handlers.

    Maps (Provider, ModeType) tuples to their handler implementations.
    Supports lazy loading and dynamic registration.

    Example:
        >>> registry.register(
        ...     provider=Provider.ANTHROPIC,
        ...     mode_type=ModeType.TOOLS,
        ...     request_handler=handle_request,
        ...     reask_handler=handle_reask,
        ...     response_parser=parse_response,
        ... )
        >>> handlers = registry.get_handlers(Provider.ANTHROPIC, ModeType.TOOLS)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._handlers: dict[Mode, ModeHandlers] = {}
        self._lazy_loaders: dict[Mode, Callable[[], ModeHandlers]] = {}

    def register(
        self,
        provider: Provider,
        mode_type: ModeType,
        request_handler: RequestHandler,
        reask_handler: ReaskHandler,
        response_parser: ResponseParser,
    ) -> None:
        """Register handlers for a mode.

        Args:
            provider: Provider enum value
            mode_type: Mode type enum value
            request_handler: Handler to prepare request kwargs
            reask_handler: Handler to handle validation failures
            response_parser: Handler to parse responses

        Raises:
            ValueError: If mode is already registered
        """
        mode = (provider, mode_type)
        if mode in self._handlers:
            raise ValueError(f"Mode {mode} is already registered")

        self._handlers[mode] = ModeHandlers(
            request_handler=request_handler,
            reask_handler=reask_handler,
            response_parser=response_parser,
        )

    def register_lazy(
        self,
        provider: Provider,
        mode_type: ModeType,
        loader: Callable[[], ModeHandlers],
    ) -> None:
        """Register a lazy loader for a mode.

        The loader will be called on first access to load handlers.

        Args:
            provider: Provider enum value
            mode_type: Mode type enum value
            loader: Callable that returns ModeHandlers when invoked

        Raises:
            ValueError: If mode is already registered
        """
        mode = (provider, mode_type)
        if mode in self._handlers or mode in self._lazy_loaders:
            raise ValueError(f"Mode {mode} is already registered")

        self._lazy_loaders[mode] = loader

    def get_handlers(self, provider: Provider, mode_type: ModeType) -> ModeHandlers:
        """Get all handlers for a mode.

        Args:
            provider: Provider enum value
            mode_type: Mode type enum value

        Returns:
            ModeHandlers with all handler functions

        Raises:
            KeyError: If mode is not registered
        """
        mode = (provider, mode_type)

        # Check if already loaded
        if mode in self._handlers:
            return self._handlers[mode]

        # Try lazy loading
        if mode in self._lazy_loaders:
            loader = self._lazy_loaders.pop(mode)
            handlers = loader()
            self._handlers[mode] = handlers
            return handlers

        raise KeyError(
            f"Mode {mode} is not registered. "
            f"Available modes: {list(self._handlers.keys())}"
        )

    def get_handler(
        self,
        provider: Provider,
        mode_type: ModeType,
        handler_type: str,
    ) -> RequestHandler | ReaskHandler | ResponseParser:
        """Get a specific handler for a mode.

        Args:
            provider: Provider enum value
            mode_type: Mode type enum value
            handler_type: One of 'request', 'reask', 'response'

        Returns:
            The requested handler function

        Raises:
            KeyError: If mode is not registered
            ValueError: If handler_type is invalid
        """
        handlers = self.get_handlers(provider, mode_type)

        if handler_type == "request":
            return handlers.request_handler
        elif handler_type == "reask":
            return handlers.reask_handler
        elif handler_type == "response":
            return handlers.response_parser
        else:
            raise ValueError(
                f"Invalid handler_type: {handler_type}. "
                f"Must be 'request', 'reask', or 'response'"
            )

    def is_registered(self, provider: Provider, mode_type: ModeType) -> bool:
        """Check if a mode is registered.

        Args:
            provider: Provider enum value
            mode_type: Mode type enum value

        Returns:
            True if mode is registered (eagerly or lazily)
        """
        mode = (provider, mode_type)
        return mode in self._handlers or mode in self._lazy_loaders

    def get_modes_for_provider(self, provider: Provider) -> list[ModeType]:
        """Get all registered mode types for a provider.

        Args:
            provider: Provider enum value

        Returns:
            List of ModeType values supported by this provider
        """
        modes = []
        for p, mt in self._handlers.keys():
            if p == provider:
                modes.append(mt)
        for p, mt in self._lazy_loaders.keys():
            if p == provider:
                modes.append(mt)
        return sorted(set(modes), key=lambda m: m.value)

    def get_providers_for_mode(self, mode_type: ModeType) -> list[Provider]:
        """Get all providers that support a mode type.

        Args:
            mode_type: Mode type enum value

        Returns:
            List of Provider values that support this mode type
        """
        providers = []
        for p, mt in self._handlers.keys():
            if mt == mode_type:
                providers.append(p)
        for p, mt in self._lazy_loaders.keys():
            if mt == mode_type:
                providers.append(p)
        return sorted(set(providers), key=lambda p: p.value)

    def list_modes(self) -> list[Mode]:
        """List all registered modes.

        Returns:
            List of (Provider, ModeType) tuples
        """
        all_modes = set(self._handlers.keys()) | set(self._lazy_loaders.keys())
        return sorted(all_modes, key=lambda m: (m[0].value, m[1].value))


# Global registry instance
mode_registry = ModeRegistry()

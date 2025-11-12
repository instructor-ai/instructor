"""Mode handler registry for v2.

Central registry mapping Mode enum values to their handler implementations.
Supports lazy loading, dynamic registration, and queryable API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from instructor import Mode, Provider

from instructor.v2.core.protocols import ReaskHandler, RequestHandler, ResponseParser


@dataclass
class ModeHandlers:
    """Collection of handlers for a specific mode."""

    request_handler: RequestHandler
    reask_handler: ReaskHandler
    response_parser: ResponseParser


class ModeRegistry:
    """Central registry for mode handlers.

    Maps (Provider, Mode) tuples to their handler implementations.
    Supports lazy loading and dynamic registration.

    Example:
        >>> registry.register(
        ...     provider=Provider.ANTHROPIC,
        ...     mode=Mode.TOOLS,
        ...     request_handler=handle_request,
        ...     reask_handler=handle_reask,
        ...     response_parser=parse_response,
        ... )
        >>> # Preferred: get all handlers at once
        >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
        >>> handlers.request_handler(...)
        >>> handlers.reask_handler(...)
        >>> handlers.response_parser(...)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._handlers: dict[Mode, ModeHandlers] = {}
        self._lazy_loaders: dict[Mode, Callable[[], ModeHandlers]] = {}

    def register(
        self,
        provider: Provider,
        mode: Mode,
        request_handler: RequestHandler,
        reask_handler: ReaskHandler,
        response_parser: ResponseParser,
    ) -> None:
        """Register handlers for a mode.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            request_handler: Handler to prepare request kwargs
            reask_handler: Handler to handle validation failures
            response_parser: Handler to parse responses

        Raises:
            ConfigurationError: If mode is already registered
        """
        from ..core.exceptions import ConfigurationError

        mode = (provider, mode)
        if mode in self._handlers:
            raise ConfigurationError(f"Mode {mode} is already registered")

        self._handlers[mode] = ModeHandlers(
            request_handler=request_handler,
            reask_handler=reask_handler,
            response_parser=response_parser,
        )

    def register_lazy(
        self,
        provider: Provider,
        mode: Mode,
        loader: Callable[[], ModeHandlers],
    ) -> None:
        """Register a lazy loader for a mode.

        The loader will be called on first access to load handlers.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            loader: Callable that returns ModeHandlers when invoked

        Raises:
            ConfigurationError: If mode is already registered
        """
        from ..core.exceptions import ConfigurationError

        mode = (provider, mode)
        if mode in self._handlers or mode in self._lazy_loaders:
            raise ConfigurationError(f"Mode {mode} is already registered")

        self._lazy_loaders[mode] = loader

    def get_handlers(self, provider: Provider, mode: Mode) -> ModeHandlers:
        """Get all handlers for a mode.

        This is the preferred method for retrieving handlers. It performs
        a single registry lookup and returns all handlers at once, which is
        more efficient than calling get_handler() multiple times.

        Args:
            provider: Provider enum value
            mode: Mode enum value

        Returns:
            ModeHandlers with all handler functions (request_handler,
            reask_handler, response_parser)

        Raises:
            KeyError: If mode is not registered

        Example:
            >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
            >>> handlers.request_handler(...)
            >>> handlers.reask_handler(...)
            >>> handlers.response_parser(...)
        """
        mode = (provider, mode)

        # Check if already loaded
        if mode in self._handlers:
            return self._handlers[mode]

        # Try lazy loading
        if mode in self._lazy_loaders:
            loader = self._lazy_loaders.pop(mode)
            handlers = loader()
            self._handlers[mode] = handlers
            return handlers

        from ..core.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Mode {mode} is not registered. "
            f"Available modes: {list(self._handlers.keys())}"
        )

    def get_handler(
        self,
        provider: Provider,
        mode: Mode,
        handler_type: str,
    ) -> RequestHandler | ReaskHandler | ResponseParser:
        """Get a specific handler for a mode.

        This is a convenience method that internally calls get_handlers().
        For better performance when you need multiple handlers, use
        get_handlers() instead and access handlers via the returned object.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            handler_type: One of 'request', 'reask', 'response'

        Returns:
            The requested handler function

        Raises:
            KeyError: If mode is not registered
            ValueError: If handler_type is invalid

        Example:
            >>> # Prefer this when you need multiple handlers:
            >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
            >>> handlers.request_handler(...)
            >>> handlers.reask_handler(...)

            >>> # Or use this convenience method for a single handler:
            >>> handler = registry.get_handler(Provider.ANTHROPIC, Mode.TOOLS, "request")
        """
        handlers = self.get_handlers(provider, mode)

        if handler_type == "request":
            return handlers.request_handler
        elif handler_type == "reask":
            return handlers.reask_handler
        elif handler_type == "response":
            return handlers.response_parser
        else:
            from ..core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid handler_type: {handler_type}. "
                f"Must be 'request', 'reask', or 'response'"
            )

    def is_registered(self, provider: Provider, mode: Mode) -> bool:
        """Check if a mode is registered.

        Args:
            provider: Provider enum value
            mode: Mode enum value

        Returns:
            True if mode is registered (eagerly or lazily)
        """
        mode = (provider, mode)
        return mode in self._handlers or mode in self._lazy_loaders

    def get_modes_for_provider(self, provider: Provider) -> list[Mode]:
        """Get all registered modes for a provider.

        Args:
            provider: Provider enum value

        Returns:
            List of Mode values supported by this provider
        """
        modes = []
        for p, mt in self._handlers.keys():
            if p == provider:
                modes.append(mt)
        for p, mt in self._lazy_loaders.keys():
            if p == provider:
                modes.append(mt)
        return sorted(set(modes), key=lambda m: m.value)

    def get_providers_for_mode(self, mode: Mode) -> list[Provider]:
        """Get all providers that support a mode.

        Args:
            mode: Mode enum value

        Returns:
            List of Provider values that support this mode
        """
        providers = []
        for p, mt in self._handlers.keys():
            if mt == mode:
                providers.append(p)
        for p, mt in self._lazy_loaders.keys():
            if mt == mode:
                providers.append(p)
        return sorted(set(providers), key=lambda p: p.value)

    def list_modes(self) -> list[Mode]:
        """List all registered modes.

        Returns:
            List of (Provider, Mode) tuples
        """
        all_modes = set(self._handlers.keys()) | set(self._lazy_loaders.keys())
        return sorted(all_modes, key=lambda m: (m[0].value, m[1].value))


# Global registry instance
mode_registry = ModeRegistry()

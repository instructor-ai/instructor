"""Handler registry for managing mode-specific response handlers."""

from typing import Any, Callable, Optional, TypeVar

from instructor.mode import Mode

T = TypeVar("T")
HandlerFunc = Callable[[type[T], dict[str, Any]], tuple[type[T], dict[str, Any]]]


class HandlerRegistry:
    """Registry for mode-specific response handlers."""

    def __init__(self) -> None:
        self._handlers: dict[Mode, HandlerFunc] = {}

    def register(self, mode: Mode, handler: HandlerFunc) -> None:
        """Register a handler for a specific mode.

        Args:
            mode: The mode to register the handler for
            handler: The handler function to register
        """
        self._handlers[mode] = handler

    def get_handler(self, mode: Mode) -> Optional[HandlerFunc]:
        """Get the handler for a specific mode.

        Args:
            mode: The mode to get the handler for

        Returns:
            The handler function if registered, None otherwise
        """
        return self._handlers.get(mode)

    def handle(
        self, mode: Mode, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle a response using the registered handler for the mode.

        Args:
            mode: The mode to use for handling
            response_model: The response model type
            kwargs: Additional keyword arguments

        Returns:
            Tuple of processed response model and updated kwargs

        Raises:
            ValueError: If no handler is registered for the mode
        """
        handler = self.get_handler(mode)
        if handler is None:
            raise ValueError(f"No handler registered for mode: {mode}")

        # If handler is an instance with a handle method, call that
        if hasattr(handler, "handle"):
            return handler.handle(response_model, kwargs)
        else:
            # Otherwise assume it's a callable
            return handler(response_model, kwargs)

    def is_registered(self, mode: Mode) -> bool:
        """Check if a handler is registered for a mode.

        Args:
            mode: The mode to check

        Returns:
            True if a handler is registered, False otherwise
        """
        return mode in self._handlers

    def register_multiple(self, handlers: dict[Mode, HandlerFunc]) -> None:
        """Register multiple handlers at once.

        Args:
            handlers: Dictionary mapping modes to handler functions
        """
        for mode, handler in handlers.items():
            self.register(mode, handler)


# Global registry instance
handler_registry = HandlerRegistry()

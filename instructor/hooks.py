from __future__ import annotations
from enum import Enum
from collections import defaultdict
from typing import Any, Literal, TypeVar, Protocol, Union

import traceback
import warnings

T = TypeVar("T")


class HookName(Enum):
    COMPLETION_KWARGS = "completion:kwargs"
    COMPLETION_RESPONSE = "completion:response"
    COMPLETION_ERROR = "completion:error"
    COMPLETION_LAST_ATTEMPT = "completion:last_attempt"
    PARSE_ERROR = "parse:error"


# Handler protocol types for type safety
class CompletionKwargsHandler(Protocol):
    """Protocol for completion kwargs handlers."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        ...


class CompletionResponseHandler(Protocol):
    """Protocol for completion response handlers."""

    def __call__(self, response: Any) -> None:
        ...


class CompletionErrorHandler(Protocol):
    """Protocol for completion error and last attempt handlers."""

    def __call__(self, error: Exception) -> None:
        ...


class ParseErrorHandler(Protocol):
    """Protocol for parse error handlers."""

    def __call__(self, error: Exception) -> None:
        ...


# Type alias for hook name parameter
HookNameType = Union[
    HookName,
    Literal[
        "completion:kwargs",
        "completion:response",
        "completion:error",
        "completion:last_attempt",
        "parse:error",
    ],
]

# Type alias for all handler types
HandlerType = Union[
    CompletionKwargsHandler,
    CompletionResponseHandler,
    CompletionErrorHandler,
    ParseErrorHandler,
]


class Hooks:
    """
    Hooks class for handling and emitting events related to completion processes.

    This class provides a mechanism to register event handlers and emit events
    for various stages of the completion process.
    """

    def __init__(self) -> None:
        """Initialize the hooks container."""
        self._handlers: defaultdict[HookName, list[HandlerType]] = defaultdict(list)

    def on(
        self,
        hook_name: HookNameType,
        handler: HandlerType,
    ) -> None:
        """
        Register an event handler for a specific event.

        This method allows you to attach a handler function to a specific event.
        When the event is emitted, all registered handlers for that event will be called.

        Args:
            hook_name: The event to listen for. This can be either a HookName enum
                       value or a string representation of the event name.
            handler: The function to be called when the event is emitted.

        Raises:
            ValueError: If the hook_name is not a valid HookName enum or string representation.

        Example:
            >>> def on_completion_kwargs(*args: Any, **kwargs: Any) -> None:
            ...     print(f"Completion kwargs: {args}, {kwargs}")
            >>> hooks = Hooks()
            >>> hooks.on(HookName.COMPLETION_KWARGS, on_completion_kwargs)
            >>> hooks.emit_completion_arguments(model="gpt-3.5-turbo", temperature=0.7)
            Completion kwargs: (), {'model': 'gpt-3.5-turbo', 'temperature': 0.7}
        """
        hook_name = self.get_hook_name(hook_name)
        self._handlers[hook_name].append(handler)

    def get_hook_name(self, hook_name: HookNameType) -> HookName:
        """
        Convert a string hook name to its corresponding enum value.

        Args:
            hook_name: Either a HookName enum value or string representation.

        Returns:
            The corresponding HookName enum value.

        Raises:
            ValueError: If the string doesn't match any HookName enum value.
        """
        if isinstance(hook_name, str):
            try:
                return HookName(hook_name)
            except ValueError as err:
                raise ValueError(f"Invalid hook name: {hook_name}") from err
        return hook_name

    def emit(self, hook_name: HookName, *args: Any, **kwargs: Any) -> None:
        """
        Generic method to emit events for any hook type.

        Args:
            hook_name: The hook to emit
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        for handler in self._handlers[hook_name]:
            try:
                handler(*args, **kwargs)  # type: ignore
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in {hook_name.value} handler:\n{error_traceback}",
                    stacklevel=2,
                )

    def emit_completion_arguments(self, *args: Any, **kwargs: Any) -> None:
        """
        Emit a completion arguments event.

        Args:
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        self.emit(HookName.COMPLETION_KWARGS, *args, **kwargs)

    def emit_completion_response(self, response: Any) -> None:
        """
        Emit a completion response event.

        Args:
            response: The completion response to pass to handlers
        """
        self.emit(HookName.COMPLETION_RESPONSE, response)

    def emit_completion_error(self, error: Exception) -> None:
        """
        Emit a completion error event.

        Args:
            error: The exception to pass to handlers
        """
        self.emit(HookName.COMPLETION_ERROR, error)

    def emit_completion_last_attempt(self, error: Exception) -> None:
        """
        Emit a completion last attempt event.

        Args:
            error: The exception to pass to handlers
        """
        self.emit(HookName.COMPLETION_LAST_ATTEMPT, error)

    def emit_parse_error(self, error: Exception) -> None:
        """
        Emit a parse error event.

        Args:
            error: The exception to pass to handlers
        """
        self.emit(HookName.PARSE_ERROR, error)

    def off(
        self,
        hook_name: HookNameType,
        handler: HandlerType,
    ) -> None:
        """
        Remove a specific handler from an event.

        Args:
            hook_name: The name of the hook.
            handler: The handler to remove.
        """
        hook_name = self.get_hook_name(hook_name)
        if hook_name in self._handlers:
            if handler in self._handlers[hook_name]:
                self._handlers[hook_name].remove(handler)
                if not self._handlers[hook_name]:
                    del self._handlers[hook_name]

    def clear(
        self,
        hook_name: HookNameType | None = None,
    ) -> None:
        """
        Clear handlers for a specific event or all events.

        Args:
            hook_name: The name of the event to clear handlers for.
                      If None, all handlers are cleared.
        """
        if hook_name is not None:
            hook_name = self.get_hook_name(hook_name)
            self._handlers.pop(hook_name, None)
        else:
            self._handlers.clear()

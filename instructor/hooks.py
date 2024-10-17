from __future__ import annotations
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, Literal, TypeVar

import traceback
import warnings

T = TypeVar("T")


class HookName(Enum):
    COMPLETION_KWARGS = "completion:kwargs"
    COMPLETION_RESPONSE = "completion:response"
    COMPLETION_ERROR = "completion:error"
    COMPLETION_LAST_ATTEMPT = "completion:last_attempt"
    PARSE_ERROR = "parse:error"


class Hooks:
    """
    Hooks class for handling and emitting events related to completion processes.

    This class provides a mechanism to register event handlers and emit events
    for various stages of the completion process.
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[HookName, list[Callable[[Any], None]]] = (
            defaultdict(list)
        )

    def on(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        """
        Registers an event handler for a specific event.

        This method allows you to attach a handler function to a specific event.
        When the event is emitted, all registered handlers for that event will be called.

        Args:
            hook_name (HookName | str): The event to listen for. This can be either a HookName enum
                                        value or a string representation of the event name.
            handler (Callable[[Any], None]): The function to be called when the event is emitted.
                                             This function should accept any arguments and return None.

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

    def get_hook_name(self, hook_name: HookName | str) -> HookName:
        if isinstance(hook_name, str):
            try:
                return HookName(hook_name)
            except ValueError as err:
                raise ValueError(f"Invalid hook name: {hook_name}") from err
        return hook_name

    def emit_completion_arguments(self, *args: Any, **kwargs: Any) -> None:
        for handler in self._handlers[HookName.COMPLETION_KWARGS]:
            try:
                handler(*args, **kwargs)
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in completion arguments handler:\n{error_traceback}",
                    stacklevel=2,
                )

    def emit_completion_response(self, response: Any) -> None:
        for handler in self._handlers[HookName.COMPLETION_RESPONSE]:
            try:
                handler(response)
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in completion response handler:\n{error_traceback}",
                    stacklevel=2,
                )

    def emit_completion_error(self, error: Exception) -> None:
        for handler in self._handlers[HookName.COMPLETION_ERROR]:
            try:
                handler(error)
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in completion error handler:\n{error_traceback}",
                    stacklevel=2,
                )

    def emit_completion_last_attempt(self, error: Exception) -> None:
        for handler in self._handlers[HookName.COMPLETION_LAST_ATTEMPT]:
            try:
                handler(error)
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in completion last attempt handler:\n{error_traceback}",
                    stacklevel=2,
                )

    def emit_parse_error(self, error: Exception) -> None:
        for handler in self._handlers[HookName.PARSE_ERROR]:
            try:
                handler(error)
            except Exception:
                error_traceback = traceback.format_exc()
                warnings.warn(
                    f"Error in parse error handler:\n{error_traceback}", stacklevel=2
                )

    def off(
        self,
        hook_name: HookName
        | Literal[
            "completion:kwargs",
            "completion:response",
            "completion:error",
            "completion:last_attempt",
            "parse:error",
        ],
        handler: Callable[[Any], None],
    ) -> None:
        """
        Removes a specific handler from an event.

        Args:
            hook_name (HookName): The name of the hook.
            handler (Callable[[Any], None]): The handler to remove.
        """
        hook_name = self.get_hook_name(hook_name)
        if hook_name in self._handlers:
            self._handlers[hook_name].remove(handler)
            if not self._handlers[hook_name]:
                del self._handlers[hook_name]

    def clear(
        self,
        hook_name: HookName
        | Literal[
            "completion:kwargs",
            "completion:response",
            "completion:error",
            "completion:last_attempt",
            "parse:error",
        ]
        | None = None,
    ) -> None:
        """
        Clears handlers for a specific event or all events.

        Args:
            hook_name (HookName | None): The name of the event to clear handlers for. If None, all handlers are cleared.
        """
        if hook_name is not None:
            hook_name = self.get_hook_name(hook_name)
            self._handlers.pop(hook_name, None)
        else:
            self._handlers.clear()

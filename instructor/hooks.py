from __future__ import annotations
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, List, Literal, TypeVar
from warnings import warn

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
    for various stages of the completion process. It supports the following events:

    - COMPLETION_KWARGS (completion:kwargs):
      Emitted when completion arguments are provided.
      Arguments: *args, **kwargs (Any arguments passed to the completion function)

    - COMPLETION_RESPONSE (completion:response):
      Emitted when a completion response is received.
      Arguments: response (The response object from the completion API)

    - COMPLETION_ERROR (completion:error):
      Emitted when an error occurs during completion.
      Arguments: error (The exception object that was raised)

    - COMPLETION_LAST_ATTEMPT (completion:last_attempt):
      Emitted on the last retry attempt.
      Arguments: error (RetryError object containing information about the failed attempts)

    - PARSE_ERROR (parse:error):
      Emitted when a parse error occurs.
      Arguments: error (The exception object that was raised)

    Usage:
        hooks = Hooks()
        hooks.on(HookName.COMPLETION_KWARGS, lambda *args, **kwargs: print("Kwargs:", kwargs))
        hooks.emit_completion_arguments(model="gpt-3.5-turbo", temperature=0.7)
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[HookName, List[Callable[[Any], None]]] = (
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
        if isinstance(hook_name, str):
            try:
                hook_name = HookName(hook_name)
            except ValueError:
                raise ValueError(f"Invalid hook name: {hook_name}")
        elif not isinstance(hook_name, HookName):
            raise ValueError(f"Invalid hook name type: {type(hook_name)}")
        self._handlers[hook_name].append(handler)

    def emit_completion_arguments(self, *args: Any, **kwargs: Any) -> None:
        for handler in self._handlers[HookName.COMPLETION_KWARGS]:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                import warnings

                warnings.warn(f"Error in completion arguments handler: {str(e)}")

    def emit_completion_response(self, response: Any) -> None:
        for handler in self._handlers[HookName.COMPLETION_RESPONSE]:
            try:
                handler(response)
            except Exception as e:
                import warnings

                warnings.warn(f"Error in completion response handler: {str(e)}")

    def emit_completion_error(self, error: Exception) -> None:
        for handler in self._handlers[HookName.COMPLETION_ERROR]:
            try:
                handler(error)
            except Exception as e:
                import warnings

                warnings.warn(f"Error in completion error handler: {str(e)}")

    def emit_completion_last_attempt(self, error: Exception) -> None:
        for handler in self._handlers[HookName.COMPLETION_LAST_ATTEMPT]:
            try:
                handler(error)
            except Exception as e:
                import warnings

                warnings.warn(f"Error in completion last attempt handler: {str(e)}")

    def emit_parse_error(self, error: Exception) -> None:
        for handler in self._handlers[HookName.PARSE_ERROR]:
            try:
                handler(error)
            except Exception as e:
                import warnings

                warnings.warn(f"Error in parse error handler: {str(e)}")

    def off(self, hook_name: HookName, handler: Callable[[Any], None]) -> None:
        """
        Removes a specific handler from an event.

        Args:
            hook_name (HookName): The name of the hook.
            handler (Callable[[Any], None]): The handler to remove.
        """
        if hook_name in self._handlers:
            self._handlers[hook_name].remove(handler)
            if not self._handlers[hook_name]:
                del self._handlers[hook_name]

    def clear(self, hook_name: HookName | None = None) -> None:
        """
        Clears handlers for a specific event or all events.

        Args:
            hook_name (HookName | None): The name of the event to clear handlers for. If None, all handlers are cleared.
        """
        if hook_name is not None:
            self._handlers.pop(hook_name, None)
        else:
            self._handlers.clear()

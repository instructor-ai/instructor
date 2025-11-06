from __future__ import annotations
from enum import Enum
from collections import defaultdict
from typing import Any, Literal, TypeVar, Protocol, Union
from dataclasses import dataclass, field
from time import time
import asyncio
import inspect

import traceback
import warnings

T = TypeVar("T")


# ============================================================================
# Context Objects - Rich metadata for hooks
# ============================================================================


@dataclass
class HookContext:
    """
    Base context object containing metadata about the current request.

    This context is passed to all hooks and provides essential information
    about the request lifecycle, retry attempts, and configuration.

    Attributes:
        request_id: Unique identifier for this request
        attempt_number: Current retry attempt (1-indexed)
        total_attempts: Maximum number of retry attempts
        is_retry: Whether this is a retry attempt (False on first attempt)
        start_time: Unix timestamp when the request started
        mode: The Mode enum value being used (e.g., Mode.TOOLS)
        response_model: The Pydantic model class expected for the response
    """
    request_id: str
    attempt_number: int
    total_attempts: int
    is_retry: bool
    start_time: float
    mode: Any  # Mode enum, using Any to avoid circular import
    response_model: type[Any] | None


@dataclass
class CompletionKwargsContext:
    """
    Context for completion:kwargs hook.

    Contains the arguments being passed to the LLM API call.

    Attributes:
        context: Base HookContext with request metadata
        args: Positional arguments passed to the API call
        kwargs: Keyword arguments passed to the API call (model, messages, etc.)
    """
    context: HookContext
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class CompletionResponseContext:
    """
    Context for completion:response hook.

    Contains the raw response from the LLM API.

    Attributes:
        context: Base HookContext with request metadata
        response: The raw response object from the API
        duration: Time taken for the API call in seconds
    """
    context: HookContext
    response: Any
    duration: float


@dataclass
class ErrorContext:
    """
    Context for error-related hooks (completion:error, parse:error).

    Contains rich information about what failed and why.

    Attributes:
        context: Base HookContext with request metadata
        error: The exception that occurred
        kwargs: The kwargs that were used for the failed request
        response: Partial response if available (may be None)
        failed_attempts: List of all previous failed attempts
        stack_trace: Formatted traceback string
    """
    context: HookContext
    error: Exception
    kwargs: dict[str, Any]
    response: Any | None
    failed_attempts: list[Any]
    stack_trace: str = field(default_factory=lambda: traceback.format_exc())


@dataclass
class ValidationContext:
    """
    Context for validation:success hook.

    Contains information about successful validation.

    Attributes:
        context: Base HookContext with request metadata
        response: The raw response from the API
        parsed_model: The successfully validated Pydantic model instance
    """
    context: HookContext
    response: Any
    parsed_model: Any


@dataclass
class RetryContext:
    """
    Context for retry:attempt hook.

    Contains information about a retry attempt.

    Attributes:
        context: Base HookContext with request metadata
        last_error: The error that triggered this retry
        next_kwargs: The modified kwargs for the next attempt
    """
    context: HookContext
    last_error: Exception
    next_kwargs: dict[str, Any]


@dataclass
class StreamChunkContext:
    """
    Context for stream:chunk hook.

    Contains information about a streaming chunk.

    Attributes:
        context: Base HookContext with request metadata
        chunk: The streaming chunk data
        chunk_index: Index of this chunk in the stream (0-indexed)
    """
    context: HookContext
    chunk: Any
    chunk_index: int


# ============================================================================
# Hook Name Enum - All available hook events
# ============================================================================


class HookName(Enum):
    """
    Enum defining all available hook events in the request lifecycle.

    Lifecycle order:
    1. REQUEST_START - Before any processing begins
    2. COMPLETION_KWARGS - Before API call (may repeat on retries)
    3. COMPLETION_RESPONSE - After successful API response
    4. STREAM_CHUNK - Each chunk in streaming mode
    5. VALIDATION_SUCCESS - After successful Pydantic validation
    6. PARSE_ERROR - When validation/parsing fails
    7. COMPLETION_ERROR - When API call fails
    8. RETRY_ATTEMPT - Before retrying after an error
    9. COMPLETION_LAST_ATTEMPT - On final retry attempt failure
    10. REQUEST_END - After request completes (success or failure)
    """
    # Request lifecycle
    REQUEST_START = "request:start"
    REQUEST_END = "request:end"

    # Completion hooks
    COMPLETION_KWARGS = "completion:kwargs"
    COMPLETION_RESPONSE = "completion:response"
    COMPLETION_ERROR = "completion:error"
    COMPLETION_LAST_ATTEMPT = "completion:last_attempt"

    # Validation hooks
    PARSE_ERROR = "parse:error"
    VALIDATION_SUCCESS = "validation:success"

    # Retry hooks
    RETRY_ATTEMPT = "retry:attempt"

    # Streaming hooks
    STREAM_CHUNK = "stream:chunk"


# ============================================================================
# Handler Protocols - Type signatures for hook handlers
# ============================================================================


class CompletionKwargsHandler(Protocol):
    """Protocol for completion:kwargs hook handlers."""
    def __call__(self, ctx: CompletionKwargsContext) -> None: ...


class CompletionResponseHandler(Protocol):
    """Protocol for completion:response hook handlers."""
    def __call__(self, ctx: CompletionResponseContext) -> None: ...


class ErrorHandler(Protocol):
    """Protocol for error hook handlers (completion:error, parse:error, completion:last_attempt)."""
    def __call__(self, ctx: ErrorContext) -> None: ...


class ValidationSuccessHandler(Protocol):
    """Protocol for validation:success hook handlers."""
    def __call__(self, ctx: ValidationContext) -> None: ...


class RetryAttemptHandler(Protocol):
    """Protocol for retry:attempt hook handlers."""
    def __call__(self, ctx: RetryContext) -> None: ...


class StreamChunkHandler(Protocol):
    """Protocol for stream:chunk hook handlers."""
    def __call__(self, ctx: StreamChunkContext) -> None: ...


class RequestLifecycleHandler(Protocol):
    """Protocol for request:start and request:end hook handlers."""
    def __call__(self, ctx: HookContext) -> None: ...


# Backward compatibility protocols (old-style handlers without context)
class LegacyCompletionKwargsHandler(Protocol):
    """Legacy protocol for old-style completion kwargs handlers."""
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class LegacyCompletionResponseHandler(Protocol):
    """Legacy protocol for old-style response handlers."""
    def __call__(self, response: Any) -> None: ...


class LegacyErrorHandler(Protocol):
    """Legacy protocol for old-style error handlers."""
    def __call__(self, error: Exception) -> None: ...


# Type alias for hook name parameter
HookNameType = Union[
    HookName,
    Literal[
        "request:start",
        "request:end",
        "completion:kwargs",
        "completion:response",
        "completion:error",
        "completion:last_attempt",
        "parse:error",
        "validation:success",
        "retry:attempt",
        "stream:chunk",
    ],
]

# Type alias for all handler types
HandlerType = Union[
    CompletionKwargsHandler,
    CompletionResponseHandler,
    ErrorHandler,
    ValidationSuccessHandler,
    RetryAttemptHandler,
    StreamChunkHandler,
    RequestLifecycleHandler,
    # Legacy handlers for backward compatibility
    LegacyCompletionKwargsHandler,
    LegacyCompletionResponseHandler,
    LegacyErrorHandler,
]


class Hooks:
    """
    Hooks class for handling and emitting events related to completion processes.

    This class provides a mechanism to register event handlers and emit events
    for various stages of the completion process.

    Features:
    - Context objects with rich metadata
    - Async handler support
    - Handler priorities for execution order
    - Backward compatibility with legacy handlers
    - Multiple handlers per event
    - Hook combination and composition
    """

    def __init__(self) -> None:
        """Initialize the hooks container."""
        # Store handlers as (priority, handler) tuples, sorted by priority descending
        self._handlers: defaultdict[HookName, list[tuple[int, HandlerType]]] = defaultdict(list)

    def on(
        self,
        hook_name: HookNameType,
        handler: HandlerType,
        priority: int = 0,
    ) -> None:
        """
        Register an event handler for a specific event.

        This method allows you to attach a handler function to a specific event.
        When the event is emitted, all registered handlers for that event will be called
        in priority order (higher priority = earlier execution).

        Args:
            hook_name: The event to listen for. This can be either a HookName enum
                       value or a string representation of the event name.
            handler: The function to be called when the event is emitted.
            priority: Execution priority (higher values execute first). Default is 0.

        Raises:
            ValueError: If the hook_name is not a valid HookName enum or string representation.

        Example:
            >>> def on_completion_kwargs(ctx: CompletionKwargsContext) -> None:
            ...     print(f"Model: {ctx.kwargs.get('model')}")
            >>> hooks = Hooks()
            >>> hooks.on(HookName.COMPLETION_KWARGS, on_completion_kwargs, priority=10)
        """
        hook_name = self.get_hook_name(hook_name)
        self._handlers[hook_name].append((priority, handler))
        # Sort by priority descending (higher priority first)
        self._handlers[hook_name].sort(key=lambda x: x[0], reverse=True)

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

    def __add__(self, other: Hooks) -> Hooks:
        """
        Combine two Hooks instances into a new one.

        This creates a new Hooks instance that contains all handlers from both
        the current instance and the other instance. Handlers are combined by
        appending the other's handlers after the current instance's handlers.

        Args:
            other: Another Hooks instance to combine with this one.

        Returns:
            A new Hooks instance containing all handlers from both instances.

        Example:
            >>> hooks1 = Hooks()
            >>> hooks2 = Hooks()
            >>> hooks1.on("completion:kwargs", lambda **kw: print("Hook 1"))
            >>> hooks2.on("completion:kwargs", lambda **kw: print("Hook 2"))
            >>> combined = hooks1 + hooks2
            >>> combined.emit_completion_arguments()  # Prints both "Hook 1" and "Hook 2"
        """
        if not isinstance(other, Hooks):
            return NotImplemented

        combined = Hooks()

        # Copy handlers from self
        for hook_name, handlers in self._handlers.items():
            combined._handlers[hook_name].extend(handlers.copy())

        # Add handlers from other
        for hook_name, handlers in other._handlers.items():
            combined._handlers[hook_name].extend(handlers.copy())

        return combined

    def __iadd__(self, other: Hooks) -> Hooks:
        """
        Add another Hooks instance to this one in-place.

        This modifies the current instance by adding all handlers from the other
        instance. The other instance's handlers are appended after the current
        instance's handlers for each event type.

        Args:
            other: Another Hooks instance to add to this one.

        Returns:
            This Hooks instance (for method chaining).

        Example:
            >>> hooks1 = Hooks()
            >>> hooks2 = Hooks()
            >>> hooks1.on("completion:kwargs", lambda **kw: print("Hook 1"))
            >>> hooks2.on("completion:kwargs", lambda **kw: print("Hook 2"))
            >>> hooks1 += hooks2
            >>> hooks1.emit_completion_arguments()  # Prints both "Hook 1" and "Hook 2"
        """
        if not isinstance(other, Hooks):
            return NotImplemented

        # Add handlers from other to self
        for hook_name, handlers in other._handlers.items():
            self._handlers[hook_name].extend(handlers.copy())

        return self

    @classmethod
    def combine(cls, *hooks_instances: Hooks) -> Hooks:
        """
        Combine multiple Hooks instances into a new one.

        This class method creates a new Hooks instance that contains all handlers
        from all provided instances. Handlers are combined in the order of the
        provided instances.

        Args:
            *hooks_instances: Variable number of Hooks instances to combine.

        Returns:
            A new Hooks instance containing all handlers from all instances.

        Example:
            >>> hooks1 = Hooks()
            >>> hooks2 = Hooks()
            >>> hooks3 = Hooks()
            >>> hooks1.on("completion:kwargs", lambda **kw: print("Hook 1"))
            >>> hooks2.on("completion:kwargs", lambda **kw: print("Hook 2"))
            >>> hooks3.on("completion:kwargs", lambda **kw: print("Hook 3"))
            >>> combined = Hooks.combine(hooks1, hooks2, hooks3)
            >>> combined.emit_completion_arguments()  # Prints all three hooks
        """
        combined = cls()

        for hooks_instance in hooks_instances:
            if not isinstance(hooks_instance, cls):
                raise TypeError(f"Expected Hooks instance, got {type(hooks_instance)}")
            combined += hooks_instance

        return combined

    def copy(self) -> Hooks:
        """
        Create a deep copy of this Hooks instance.

        Returns:
            A new Hooks instance with all the same handlers.

        Example:
            >>> original = Hooks()
            >>> original.on("completion:kwargs", lambda **kw: print("Hook"))
            >>> copy = original.copy()
            >>> copy.emit_completion_arguments()  # Prints "Hook"
        """
        new_hooks = Hooks()
        for hook_name, handlers in self._handlers.items():
            new_hooks._handlers[hook_name].extend(handlers.copy())
        return new_hooks

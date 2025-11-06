"""v2 patch mechanism using hierarchical registry.

Simplified patching logic that uses the v2 mode registry for handler dispatch.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from instructor.core.hooks import Hooks
from instructor.templating import handle_templating
from instructor.utils import is_async
from instructor.v2.core.mode_types import ModeType, Provider
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tenacity import AsyncRetrying, Retrying

logger = logging.getLogger("instructor.v2")

T_Model = TypeVar("T_Model", bound=BaseModel)


def handle_context(
    context: dict[str, Any] | None = None,
    validation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Handle context and validation_context parameters.

    Args:
        context: New-style context parameter
        validation_context: Deprecated validation_context parameter

    Returns:
        Merged context dict or None

    Raises:
        ValueError: If both parameters are provided
    """
    if context is not None and validation_context is not None:
        raise ValueError(
            "Cannot provide both 'context' and 'validation_context'. "
            "Use 'context' instead."
        )
    if validation_context is not None and context is None:
        import warnings

        warnings.warn(
            "'validation_context' is deprecated. Use 'context' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        context = validation_context
    return context


def patch_v2(
    func: Callable[..., Any],
    provider: Provider,
    mode_type: ModeType,
    default_model: str | None = None,
) -> Callable[..., T_Model]:
    """Patch a function to use v2 registry for structured outputs.

    Args:
        func: Function to patch (e.g., client.messages.create)
        provider: Provider enum value
        mode_type: Mode type enum value
        default_model: Default model to inject if not provided in request

    Returns:
        Patched function that supports response_model parameter
    """
    logger.debug(
        f"Patching with v2 registry: {provider=}, {mode_type=}, {default_model=}"
    )

    # Check if handlers are registered
    if not mode_registry.is_registered(provider, mode_type):
        raise ValueError(
            f"Mode ({provider}, {mode_type}) is not registered. "
            f"Available modes: {mode_registry.list_modes()}"
        )

    func_is_async = is_async(func)

    if func_is_async:
        return _create_async_wrapper(func, provider, mode_type, default_model)
    else:
        return _create_sync_wrapper(func, provider, mode_type, default_model)


def _create_sync_wrapper(
    func: Callable[..., Any],
    provider: Provider,
    mode_type: ModeType,
    default_model: str | None = None,
) -> Callable[..., T_Model]:
    """Create synchronous wrapper for patched function."""

    @wraps(func)
    def new_create_sync(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model:
        """Patched synchronous create function."""
        context = handle_context(context, validation_context)

        # Inject default model if not provided and available
        if default_model is not None and "model" not in kwargs:
            kwargs["model"] = default_model

        # Get request handler from registry
        request_handler = mode_registry.get_handler(provider, mode_type, "request")

        # Prepare request kwargs using registry handler
        response_model, new_kwargs = request_handler(
            response_model=response_model, kwargs=kwargs
        )

        # Handle templating
        new_kwargs = handle_templating(
            new_kwargs,
            mode=(provider, mode_type),  # Pass as tuple
            context=context,
        )

        # Use v2 retry logic with registry handlers
        response = retry_sync_v2(
            func=func,
            response_model=response_model,
            provider=provider,
            mode_type=mode_type,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            strict=strict,
            hooks=hooks,
        )

        return response  # type: ignore[return-value]

    return new_create_sync  # type: ignore[return-value]


def _create_async_wrapper(
    func: Callable[..., Awaitable[Any]],
    provider: Provider,
    mode_type: ModeType,
    default_model: str | None = None,
) -> Callable[..., Awaitable[T_Model]]:
    """Create asynchronous wrapper for patched function."""

    @wraps(func)
    async def new_create_async(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model:
        """Patched asynchronous create function."""
        context = handle_context(context, validation_context)

        # Inject default model if not provided and available
        if default_model is not None and "model" not in kwargs:
            kwargs["model"] = default_model

        # Get request handler from registry
        request_handler = mode_registry.get_handler(provider, mode_type, "request")

        # Prepare request kwargs using registry handler
        response_model, new_kwargs = request_handler(
            response_model=response_model, kwargs=kwargs
        )

        # Handle templating
        new_kwargs = handle_templating(
            new_kwargs,
            mode=(provider, mode_type),  # Pass as tuple
            context=context,
        )

        # Use v2 retry logic with registry handlers
        response = await retry_async_v2(
            func=func,
            response_model=response_model,
            provider=provider,
            mode_type=mode_type,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            strict=strict,
            hooks=hooks,
        )

        return response  # type: ignore[return-value]

    return new_create_async  # type: ignore[return-value]

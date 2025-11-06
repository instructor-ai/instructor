"""v2 retry mechanism using registry handlers.

Custom retry logic for v2 that uses registry's reask and response_parser
instead of v1's process_response.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from instructor.core.exceptions import InstructorRetryException
from instructor.v2.core.mode_types import ModeType, Provider
from instructor.v2.core.registry import mode_registry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from instructor.core.hooks import Hooks

logger = logging.getLogger("instructor.v2.retry")

T_Model = TypeVar("T_Model", bound=BaseModel)


def retry_sync_v2(
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    provider: Provider,
    mode_type: ModeType,
    context: dict[str, Any] | None,
    max_retries: int | Retrying,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    strict: bool,
    hooks: Hooks | None = None,
) -> T_Model:
    """Sync retry logic using v2 registry handlers.

    Args:
        func: API function to call
        response_model: Pydantic model to extract
        provider: Provider enum
        mode_type: ModeType enum
        context: Validation context
        max_retries: Max retry attempts or Retrying instance
        args: Positional args for func
        kwargs: Keyword args for func
        strict: Strict validation mode
        hooks: Optional hooks

    Returns:
        Validated Pydantic model instance

    Raises:
        InstructorRetryException: If max retries exceeded
    """
    if response_model is None:
        # No structured output, just call the API
        return func(*args, **kwargs)

    # Get handlers from registry
    response_parser = mode_registry.get_handler(provider, mode_type, "response")
    reask_handler = mode_registry.get_handler(provider, mode_type, "reask")

    # Setup retrying
    if isinstance(max_retries, int):
        max_retries_instance: Retrying = Retrying(
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    attempts = []
    last_exception = None

    try:
        for attempt in max_retries_instance:
            with attempt:
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                response = func(*args, **kwargs)

                if hooks:
                    hooks.emit_completion_response(response)

                # Parse response using registry
                try:
                    parsed = response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                    )

                    return parsed  # type: ignore

                except ValidationError as e:
                    logger.debug(f"Validation error: {e}")
                    attempts.append(
                        {
                            "exception": str(e),
                            "completion": response,
                        }
                    )
                    last_exception = e

                    if hooks:
                        hooks.emit_parse_error(e)

                    # Prepare reask using registry
                    kwargs = reask_handler(
                        kwargs=kwargs,
                        response=response,
                        exception=e,
                    )

                    # Will retry with modified kwargs
                    raise

    except Exception as e:
        # Max retries exceeded
        raise InstructorRetryException(
            n_attempts=len(attempts),
            messages=[a.get("exception") for a in attempts],
            last_completion=attempts[-1].get("completion") if attempts else None,
            total_usage=None,
        ) from last_exception or e

    # Should never reach here
    raise InstructorRetryException(
        n_attempts=len(attempts),
        messages=[a.get("exception") for a in attempts],
        last_completion=attempts[-1].get("completion") if attempts else None,
        total_usage=None,
    )


async def retry_async_v2(
    func: Callable[..., Awaitable[Any]],
    response_model: type[T_Model] | None,
    provider: Provider,
    mode_type: ModeType,
    context: dict[str, Any] | None,
    max_retries: int | AsyncRetrying,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    strict: bool,
    hooks: Hooks | None = None,
) -> T_Model:
    """Async retry logic using v2 registry handlers.

    Args:
        func: Async API function to call
        response_model: Pydantic model to extract
        provider: Provider enum
        mode_type: ModeType enum
        context: Validation context
        max_retries: Max retry attempts or AsyncRetrying instance
        args: Positional args for func
        kwargs: Keyword args for func
        strict: Strict validation mode
        hooks: Optional hooks

    Returns:
        Validated Pydantic model instance

    Raises:
        InstructorRetryException: If max retries exceeded
    """
    if response_model is None:
        # No structured output, just call the API
        return await func(*args, **kwargs)

    # Get handlers from registry
    response_parser = mode_registry.get_handler(provider, mode_type, "response")
    reask_handler = mode_registry.get_handler(provider, mode_type, "reask")

    # Setup retrying
    if isinstance(max_retries, int):
        max_retries_instance: AsyncRetrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    attempts = []
    last_exception = None

    try:
        async for attempt in max_retries_instance:
            with attempt:
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                response = await func(*args, **kwargs)

                if hooks:
                    hooks.emit_completion_response(response)

                # Parse response using registry
                try:
                    parsed = response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                    )

                    return parsed  # type: ignore

                except ValidationError as e:
                    logger.debug(f"Validation error: {e}")
                    attempts.append(
                        {
                            "exception": str(e),
                            "completion": response,
                        }
                    )
                    last_exception = e

                    if hooks:
                        hooks.emit_parse_error(e)

                    # Prepare reask using registry
                    kwargs = reask_handler(
                        kwargs=kwargs,
                        response=response,
                        exception=e,
                    )

                    # Will retry with modified kwargs
                    raise

    except Exception as e:
        # Max retries exceeded
        raise InstructorRetryException(
            n_attempts=len(attempts),
            messages=[a.get("exception") for a in attempts],
            last_completion=attempts[-1].get("completion") if attempts else None,
            total_usage=None,
        ) from last_exception or e

    # Should never reach here
    raise InstructorRetryException(
        n_attempts=len(attempts),
        messages=[a.get("exception") for a in attempts],
        last_completion=attempts[-1].get("completion") if attempts else None,
        total_usage=None,
    )

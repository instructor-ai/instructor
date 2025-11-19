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

from instructor import Mode, Provider
from instructor.core.exceptions import FailedAttempt, InstructorRetryException
from instructor.core.retry import extract_messages
from instructor.v2.core.exceptions import RegistryValidationMixin
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
    mode: Mode,
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
        mode: Mode enum
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

    # Validate and get handlers from registry
    RegistryValidationMixin.validate_mode_registration(provider, mode)
    handlers = mode_registry.get_handlers(provider, mode)

    # Setup retrying
    if isinstance(max_retries, int):
        max_retries_instance: Retrying = Retrying(
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    failed_attempts: list[FailedAttempt] = []
    last_exception: Exception | None = None
    total_usage = 0

    try:
        for attempt in max_retries_instance:
            with attempt:
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                try:
                    response = func(*args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"API call failed on attempt "
                        f"{attempt.retry_state.attempt_number}: {e}"
                    )
                    raise

                if hooks:
                    hooks.emit_completion_response(response)

                # Parse response using registry
                try:
                    parsed = handlers.response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                    )
                    logger.debug(
                        f"Successfully parsed response on attempt "
                        f"{attempt.retry_state.attempt_number}"
                    )
                    return parsed  # type: ignore

                except ValidationError as e:
                    attempt_number = attempt.retry_state.attempt_number
                    logger.debug(f"Validation error on attempt {attempt_number}: {e}")
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )
                    last_exception = e

                    if hooks:
                        hooks.emit_parse_error(e)

                    # Prepare reask using registry
                    kwargs = handlers.reask_handler(
                        kwargs=kwargs,
                        response=response,
                        exception=e,
                    )

                    # Will retry with modified kwargs
                    raise

    except Exception as e:
        # Max retries exceeded or non-validation error occurred
        if last_exception is None:
            last_exception = e

        logger.error(
            f"Max retries exceeded. Total attempts: {len(failed_attempts)}, "
            f"Last error: {last_exception}"
        )

        raise InstructorRetryException(
            str(last_exception),
            last_completion=failed_attempts[-1].completion if failed_attempts else None,
            n_attempts=len(failed_attempts),
            total_usage=total_usage,
            messages=extract_messages(kwargs),
            create_kwargs=kwargs,
            failed_attempts=failed_attempts,
        ) from last_exception

    # Should never reach here
    logger.error("Unexpected code path in retry_sync_v2")
    raise InstructorRetryException(
        str(last_exception) if last_exception else "Unknown error",
        last_completion=failed_attempts[-1].completion if failed_attempts else None,
        n_attempts=len(failed_attempts),
        total_usage=total_usage,
        messages=extract_messages(kwargs),
        create_kwargs=kwargs,
        failed_attempts=failed_attempts,
    )


async def retry_async_v2(
    func: Callable[..., Awaitable[Any]],
    response_model: type[T_Model] | None,
    provider: Provider,
    mode: Mode,
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
        mode: Mode enum
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

    # Validate and get handlers from registry
    RegistryValidationMixin.validate_mode_registration(provider, mode)
    handlers = mode_registry.get_handlers(provider, mode)

    # Setup retrying
    if isinstance(max_retries, int):
        max_retries_instance: AsyncRetrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    failed_attempts: list[FailedAttempt] = []
    last_exception: Exception | None = None
    total_usage = 0

    try:
        async for attempt in max_retries_instance:
            with attempt:
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                try:
                    response = await func(*args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"API call failed on attempt "
                        f"{attempt.retry_state.attempt_number}: {e}"
                    )
                    raise

                if hooks:
                    hooks.emit_completion_response(response)

                # Check if this is a streaming response
                stream = kwargs.get("stream", False)
                if stream and response_model is not None:
                    from instructor.dsl.iterable import IterableBase
                    from instructor.dsl.partial import PartialBase
                    import inspect

                    # Handle streaming responses for IterableBase and PartialBase
                    if inspect.isclass(response_model) and issubclass(
                        response_model, (IterableBase, PartialBase)
                    ):
                        # Map mode for streaming: Anthropic TOOLS mode needs ANTHROPIC_TOOLS
                        # for extract_json to work correctly (checks for Mode.ANTHROPIC_TOOLS)
                        streaming_mode = mode
                        if provider == Provider.ANTHROPIC and mode == Mode.TOOLS:
                            streaming_mode = Mode.ANTHROPIC_TOOLS
                        elif provider == Provider.ANTHROPIC and mode == Mode.JSON:
                            streaming_mode = Mode.ANTHROPIC_JSON

                        # Return the async generator directly for streaming
                        return response_model.from_streaming_response_async(  # type: ignore
                            response, mode=streaming_mode
                        )

                # Parse response using registry
                try:
                    parsed = handlers.response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                    )
                    logger.debug(
                        f"Successfully parsed response on attempt "
                        f"{attempt.retry_state.attempt_number}"
                    )
                    return parsed  # type: ignore

                except ValidationError as e:
                    attempt_number = attempt.retry_state.attempt_number
                    logger.debug(f"Validation error on attempt {attempt_number}: {e}")
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )
                    last_exception = e

                    if hooks:
                        hooks.emit_parse_error(e)

                    # Prepare reask using registry
                    kwargs = handlers.reask_handler(
                        kwargs=kwargs,
                        response=response,
                        exception=e,
                    )

                    # Will retry with modified kwargs
                    raise

    except Exception as e:
        # Max retries exceeded or non-validation error occurred
        if last_exception is None:
            last_exception = e

        logger.error(
            f"Max retries exceeded. Total attempts: {len(failed_attempts)}, "
            f"Last error: {last_exception}"
        )

        raise InstructorRetryException(
            str(last_exception),
            last_completion=failed_attempts[-1].completion if failed_attempts else None,
            n_attempts=len(failed_attempts),
            total_usage=total_usage,
            messages=extract_messages(kwargs),
            create_kwargs=kwargs,
            failed_attempts=failed_attempts,
        ) from last_exception

    # Should never reach here
    logger.error("Unexpected code path in retry_async_v2")
    raise InstructorRetryException(
        str(last_exception) if last_exception else "Unknown error",
        last_completion=failed_attempts[-1].completion if failed_attempts else None,
        n_attempts=len(failed_attempts),
        total_usage=total_usage,
        messages=extract_messages(kwargs),
        create_kwargs=kwargs,
        failed_attempts=failed_attempts,
    )

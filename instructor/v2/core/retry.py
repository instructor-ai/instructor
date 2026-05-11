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
    stop_after_delay,
)

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import (
    FailedAttempt,
    IncompleteOutputException,
    InstructorRetryException,
)
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.response_list import ListResponse
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.messages import extract_messages
from instructor.v2.core.usage import update_total_usage
from instructor.v2.core.exceptions import RegistryValidationMixin
from instructor.v2.core.registry import mode_registry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from instructor.v2.core.hooks import Hooks

logger = logging.getLogger("instructor.v2.retry")

T_Model = TypeVar("T_Model", bound=BaseModel)


def _finalize_parsed_response(parsed: Any, response: Any) -> Any:
    if isinstance(parsed, IterableBase):
        parsed = [task for task in parsed.tasks]
    if isinstance(parsed, AdapterBase):
        return parsed.content
    if isinstance(parsed, list) and not isinstance(parsed, ListResponse):
        return ListResponse.from_list(parsed, raw_response=response)
    if isinstance(parsed, BaseModel):
        parsed._raw_response = response  # type: ignore[attr-defined]
    return parsed


def _initialize_usage(provider: Provider | Mode) -> Any:
    from openai.types.completion_usage import (
        CompletionTokensDetails,
        CompletionUsage,
        PromptTokensDetails,
    )

    total_usage: Any = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    )
    anthropic_modes = {
        Mode.ANTHROPIC_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
        Mode.ANTHROPIC_JSON,
        Mode.ANTHROPIC_PARALLEL_TOOLS,
    }
    if provider is Provider.ANTHROPIC or provider in anthropic_modes:
        from instructor.v2.providers.anthropic.usage import initialize_usage

        total_usage = initialize_usage()
    return total_usage


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
        stop_condition = stop_after_attempt(max(max_retries, 1))
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)):
            stop_condition = stop_condition | stop_after_delay(timeout)
        max_retries_instance = Retrying(
            stop=stop_condition,
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    failed_attempts: list[FailedAttempt] = []
    last_exception: Exception | None = None
    total_usage = _initialize_usage(provider)

    try:
        for attempt in max_retries_instance:
            with attempt:
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                try:
                    response = func(*args, **kwargs)
                except IncompleteOutputException:
                    raise
                except Exception as e:
                    logger.error(
                        f"API call failed on attempt "
                        f"{attempt.retry_state.attempt_number}: {e}"
                    )
                    raise

                if hooks:
                    hooks.emit_completion_response(response)

                update_total_usage(response=response, total_usage=total_usage)

                # Parse response using registry
                try:
                    stream = kwargs.get("stream", False)
                    parsed = handlers.response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        stream=stream,
                        is_async=False,
                    )
                    logger.debug(
                        f"Successfully parsed response on attempt "
                        f"{attempt.retry_state.attempt_number}"
                    )
                    return _finalize_parsed_response(parsed, response)  # type: ignore

                except IncompleteOutputException:
                    raise
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

    except IncompleteOutputException:
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


def retry_sync(
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: Any,
    context: dict[str, Any] | None = None,
    max_retries: int | Retrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
    hooks: Hooks | None = None,
) -> T_Model | None:
    """Compatibility wrapper for the public retry API."""
    strict_value = True if strict is None else strict
    return retry_sync_v2(
        func=func,
        response_model=response_model,
        provider=provider,
        mode=mode,
        context=context,
        max_retries=max_retries,
        args=tuple(args) if isinstance(args, tuple) else args,
        kwargs=dict(kwargs),
        strict=strict_value,
        hooks=hooks,
    )


async def retry_async(
    func: Callable[..., Any],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: Any,
    context: dict[str, Any] | None = None,
    max_retries: int | AsyncRetrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
    hooks: Hooks | None = None,
) -> T_Model | None:
    """Compatibility wrapper for the public retry API."""
    strict_value = True if strict is None else strict
    return await retry_async_v2(
        func=func,
        response_model=response_model,
        provider=provider,
        mode=mode,
        context=context,
        max_retries=max_retries,
        args=tuple(args) if isinstance(args, tuple) else args,
        kwargs=dict(kwargs),
        strict=strict_value,
        hooks=hooks,
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
        stop_condition = stop_after_attempt(max(max_retries, 1))
        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)):
            stop_condition = stop_condition | stop_after_delay(timeout)
        max_retries_instance = AsyncRetrying(
            stop=stop_condition,
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        )
    else:
        max_retries_instance = max_retries

    failed_attempts: list[FailedAttempt] = []
    last_exception: Exception | None = None
    total_usage = _initialize_usage(provider)

    try:
        async for attempt in max_retries_instance:
            with attempt:
                # Call API
                if hooks:
                    hooks.emit_completion_arguments(**kwargs)

                try:
                    response = await func(*args, **kwargs)
                except IncompleteOutputException:
                    raise
                except Exception as e:
                    logger.error(
                        f"API call failed on attempt "
                        f"{attempt.retry_state.attempt_number}: {e}"
                    )
                    raise

                if hooks:
                    hooks.emit_completion_response(response)

                update_total_usage(response=response, total_usage=total_usage)

                # Parse response using registry
                try:
                    stream = kwargs.get("stream", False)
                    parsed = handlers.response_parser(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        stream=stream,
                        is_async=True,
                    )
                    logger.debug(
                        f"Successfully parsed response on attempt "
                        f"{attempt.retry_state.attempt_number}"
                    )
                    return _finalize_parsed_response(parsed, response)  # type: ignore

                except IncompleteOutputException:
                    raise
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

    except IncompleteOutputException:
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

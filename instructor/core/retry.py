# type: ignore[all]

from __future__ import annotations

import logging
from json import JSONDecodeError
from typing import Any, Callable, TypeVar

from .exceptions import (
    InstructorRetryException,
    AsyncValidationError,
    FailedAttempt,
    ValidationError as InstructorValidationError,
)
from .hooks import Hooks
from ..mode import Mode
from ..processing.response import (
    process_response,
    process_response_async,
    handle_reask_kwargs,
)
from ..utils import update_total_usage
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import (
    CompletionUsage,
    CompletionTokensDetails,
    PromptTokensDetails,
)
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    stop_after_attempt,
    stop_after_delay,
)
from typing_extensions import ParamSpec

logger = logging.getLogger("instructor")

# Type Variables
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def initialize_retrying(
    max_retries: int | Retrying | AsyncRetrying,
    is_async: bool,
    timeout: float | None = None,
):
    """
    Initialize the retrying mechanism based on the type (synchronous or asynchronous).

    Args:
        max_retries (int | Retrying | AsyncRetrying): Maximum number of retries or a retrying object.
        is_async (bool): Flag indicating if the retrying is asynchronous.
        timeout (float | None): Optional timeout in seconds to limit total retry duration.

    Returns:
        Retrying | AsyncRetrying: Configured retrying object.
    """
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}, timeout: {timeout}")

        # Create stop conditions
        stop_conditions = [stop_after_attempt(max_retries)]
        if timeout is not None:
            # Add global timeout: stop after timeout seconds total
            stop_conditions.append(stop_after_delay(timeout))

        # Combine stop conditions with OR logic (stop if ANY condition is met)
        stop_condition = stop_conditions[0]
        for condition in stop_conditions[1:]:
            stop_condition = stop_condition | condition

        if is_async:
            max_retries = AsyncRetrying(stop=stop_condition)
        else:
            max_retries = Retrying(stop=stop_condition)
    elif not isinstance(max_retries, (Retrying, AsyncRetrying)):
        from .exceptions import ConfigurationError

        raise ConfigurationError(
            "max_retries must be an int or a `tenacity.Retrying`/`tenacity.AsyncRetrying` object"
        )
    return max_retries


def initialize_usage(mode: Mode) -> CompletionUsage | Any:
    """
    Initialize the total usage based on the mode.

    Args:
        mode (Mode): The mode of operation.

    Returns:
        CompletionUsage | Any: Initialized usage object.
    """
    total_usage = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    )
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
        from anthropic.types import Usage as AnthropicUsage

        total_usage = AnthropicUsage(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
    return total_usage


def extract_messages(kwargs: dict[str, Any]) -> Any:
    """
    Extract messages from kwargs, helps handles the cohere and gemini chat history cases

    Args:
        kwargs (Dict[str, Any]): Keyword arguments containing message data.

    Returns:
        Any: Extracted messages.
    """
    # Directly check for keys in an efficient order (most common first)
    # instead of nested get() calls which are inefficient
    if "messages" in kwargs:
        return kwargs["messages"]
    if "contents" in kwargs:
        return kwargs["contents"]
    if "chat_history" in kwargs:
        return kwargs["chat_history"]
    return []


def retry_sync(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: Any,
    context: dict[str, Any] | None = None,
    max_retries: int | Retrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    hooks: Hooks | None = None,
) -> T_Model | None:
    """
    Retry a synchronous function upon specified exceptions.

    Args:
        func (Callable[T_ParamSpec, T_Retval]): The function to retry.
        response_model (Optional[type[T_Model]]): The model to validate the response against.
        args (Any): Positional arguments for the function.
        kwargs (Any): Keyword arguments for the function.
        context (Optional[Dict[str, Any]], optional): Additional context for validation. Defaults to None.
        max_retries (int | Retrying, optional): Maximum number of retries or a retrying object. Defaults to 1.
        strict (Optional[bool], optional): Strict mode flag. Defaults to None.
        mode (Mode, optional): The mode of operation. Defaults to Mode.TOOLS.
        hooks (Optional[Hooks], optional): Hooks for emitting events. Defaults to None.

    Returns:
        T_Model | None: The processed response model or None.

    Raises:
        InstructorRetryException: If all retry attempts fail.
    """
    hooks = hooks or Hooks()
    total_usage = initialize_usage(mode)
    # Extract timeout from kwargs if available (for global timeout across retries)
    timeout = kwargs.get("timeout")
    max_retries = initialize_retrying(max_retries, is_async=False, timeout=timeout)

    # Pre-extract stream flag to avoid repeated lookup
    stream = kwargs.get("stream", False)

    # Track all failed attempts
    failed_attempts: list[FailedAttempt] = []

    try:
        response = None
        for attempt in max_retries:
            with attempt:
                logger.debug(f"Retrying, attempt: {attempt.retry_state.attempt_number}")
                try:
                    hooks.emit_completion_arguments(*args, **kwargs)
                    response = func(*args, **kwargs)
                    hooks.emit_completion_response(response)
                    response = update_total_usage(
                        response=response, total_usage=total_usage
                    )

                    return process_response(  # type: ignore
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        mode=mode,
                        stream=stream,
                    )
                except (
                    ValidationError,
                    JSONDecodeError,
                    InstructorValidationError,
                ) as e:
                    logger.debug(f"Parse error: {e}")
                    hooks.emit_parse_error(e)

                    # Track this failed attempt
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )

                    # Check if this is the last attempt
                    if isinstance(max_retries, Retrying) and hasattr(
                        max_retries, "stop"
                    ):
                        # For tenacity Retrying objects, check if next attempt would exceed limit
                        will_retry = (
                            attempt.retry_state.outcome is None
                            or not attempt.retry_state.outcome.failed
                        )
                        is_last_attempt = (
                            not will_retry
                            or attempt.retry_state.attempt_number
                            >= getattr(
                                max_retries.stop, "max_attempt_number", float("inf")
                            )
                        )
                        if is_last_attempt:
                            hooks.emit_completion_last_attempt(e)

                    kwargs = handle_reask_kwargs(
                        kwargs=kwargs,
                        mode=mode,
                        response=response,
                        exception=e,
                        failed_attempts=failed_attempts,
                    )
                    raise e
                except Exception as e:
                    # Emit completion:error for non-validation errors (API errors, network errors, etc.)
                    logger.debug(f"Completion error: {e}")
                    hooks.emit_completion_error(e)

                    # Track this failed attempt
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )

                    # Check if this is the last attempt for completion errors
                    if isinstance(max_retries, Retrying) and hasattr(
                        max_retries, "stop"
                    ):
                        will_retry = (
                            attempt.retry_state.outcome is None
                            or not attempt.retry_state.outcome.failed
                        )
                        is_last_attempt = (
                            not will_retry
                            or attempt.retry_state.attempt_number
                            >= getattr(
                                max_retries.stop, "max_attempt_number", float("inf")
                            )
                        )
                        if is_last_attempt:
                            hooks.emit_completion_last_attempt(e)
                    raise e
    except RetryError as e:
        logger.debug(f"Retry error: {e}")
        raise InstructorRetryException(
            e.last_attempt._exception,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            #! deprecate messages soon
            messages=extract_messages(
                kwargs
            ),  # Use the optimized function instead of nested lookups
            create_kwargs=kwargs,
            total_usage=total_usage,
            failed_attempts=failed_attempts,
        ) from e


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: type[T_Model] | None,
    args: Any,
    kwargs: Any,
    context: dict[str, Any] | None = None,
    max_retries: int | AsyncRetrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    hooks: Hooks | None = None,
) -> T_Model | None:
    """
    Retry an asynchronous function upon specified exceptions.

    Args:
        func (Callable[T_ParamSpec, T_Retval]): The asynchronous function to retry.
        response_model (Optional[type[T_Model]]): The model to validate the response against.
        context (Optional[Dict[str, Any]]): Additional context for validation.
        args (Any): Positional arguments for the function.
        kwargs (Any): Keyword arguments for the function.
        max_retries (int | AsyncRetrying, optional): Maximum number of retries or an async retrying object. Defaults to 1.
        strict (Optional[bool], optional): Strict mode flag. Defaults to None.
        mode (Mode, optional): The mode of operation. Defaults to Mode.TOOLS.
        hooks (Optional[Hooks], optional): Hooks for emitting events. Defaults to None.

    Returns:
        T_Model | None: The processed response model or None.

    Raises:
        InstructorRetryException: If all retry attempts fail.
    """
    hooks = hooks or Hooks()
    total_usage = initialize_usage(mode)
    # Extract timeout from kwargs if available (for global timeout across retries)
    timeout = kwargs.get("timeout")
    max_retries = initialize_retrying(max_retries, is_async=True, timeout=timeout)

    # Pre-extract stream flag to avoid repeated lookup
    stream = kwargs.get("stream", False)

    # Track all failed attempts
    failed_attempts: list[FailedAttempt] = []

    try:
        response = None
        async for attempt in max_retries:
            logger.debug(f"Retrying, attempt: {attempt.retry_state.attempt_number}")
            with attempt:
                try:
                    hooks.emit_completion_arguments(*args, **kwargs)
                    response: ChatCompletion = await func(*args, **kwargs)
                    hooks.emit_completion_response(response)
                    response = update_total_usage(
                        response=response, total_usage=total_usage
                    )

                    return await process_response_async(
                        response=response,
                        response_model=response_model,
                        validation_context=context,
                        strict=strict,
                        mode=mode,
                        stream=stream,
                    )
                except (
                    ValidationError,
                    JSONDecodeError,
                    AsyncValidationError,
                    InstructorValidationError,
                ) as e:
                    logger.debug(f"Parse error: {e}")
                    hooks.emit_parse_error(e)

                    # Track this failed attempt
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )

                    # Check if this is the last attempt
                    if isinstance(max_retries, AsyncRetrying) and hasattr(
                        max_retries, "stop"
                    ):
                        # For tenacity AsyncRetrying objects, check if next attempt would exceed limit
                        will_retry = (
                            attempt.retry_state.outcome is None
                            or not attempt.retry_state.outcome.failed
                        )
                        is_last_attempt = (
                            not will_retry
                            or attempt.retry_state.attempt_number
                            >= getattr(
                                max_retries.stop, "max_attempt_number", float("inf")
                            )
                        )
                        if is_last_attempt:
                            hooks.emit_completion_last_attempt(e)

                    kwargs = handle_reask_kwargs(
                        kwargs=kwargs,
                        mode=mode,
                        response=response,
                        exception=e,
                        failed_attempts=failed_attempts,
                    )
                    raise e
                except Exception as e:
                    # Emit completion:error for non-validation errors (API errors, network errors, etc.)
                    logger.debug(f"Completion error: {e}")
                    hooks.emit_completion_error(e)

                    # Track this failed attempt
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt.retry_state.attempt_number,
                            exception=e,
                            completion=response,
                        )
                    )

                    # Check if this is the last attempt for completion errors
                    if isinstance(max_retries, AsyncRetrying) and hasattr(
                        max_retries, "stop"
                    ):
                        will_retry = (
                            attempt.retry_state.outcome is None
                            or not attempt.retry_state.outcome.failed
                        )
                        is_last_attempt = (
                            not will_retry
                            or attempt.retry_state.attempt_number
                            >= getattr(
                                max_retries.stop, "max_attempt_number", float("inf")
                            )
                        )
                        if is_last_attempt:
                            hooks.emit_completion_last_attempt(e)
                    raise e
    except RetryError as e:
        logger.debug(f"Retry error: {e}")
        raise InstructorRetryException(
            e.last_attempt._exception,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            #! deprecate messages soon
            messages=extract_messages(
                kwargs
            ),  # Use the optimized function instead of nested lookups
            create_kwargs=kwargs,
            total_usage=total_usage,
            failed_attempts=failed_attempts,
        ) from e

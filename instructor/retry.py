# type: ignore[all]
from __future__ import annotations

import logging
from json import JSONDecodeError
from typing import Any, Callable, TypeVar

from instructor.exceptions import InstructorRetryException
from instructor.hooks import Hooks
from instructor.mode import Mode
from instructor.reask import handle_reask_kwargs
from instructor.process_response import process_response, process_response_async
from instructor.utils import update_total_usage
from instructor.validators import AsyncValidationError
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, RetryError, Retrying, stop_after_attempt
from typing_extensions import ParamSpec

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


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
    hooks = hooks or Hooks()

    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
        from anthropic.types import Usage as AnthropicUsage

        total_usage = AnthropicUsage(input_tokens=0, output_tokens=0)

    # If max_retries is int, then create a Retrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = Retrying(
            stop=stop_after_attempt(max_retries),
        )
    if not isinstance(max_retries, (Retrying, AsyncRetrying)):
        raise ValueError("max_retries must be an int or a `tenacity.Retrying` object")

    try:
        response = None
        for attempt in max_retries:
            with attempt:
                try:
                    hooks.emit_completion_arguments(*args, **kwargs)
                    try:
                        response = func(*args, **kwargs)
                    except Exception as e:
                        hooks.emit_completion_error(e)
                        raise e
                    hooks.emit_completion_response(response)
                    stream = kwargs.get("stream", False)
                    response = update_total_usage(response, total_usage)
                    return process_response(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    hooks.emit_parse_error(e)
                    logger.debug(f"Error response: {response}")
                    kwargs = handle_reask_kwargs(
                        kwargs=kwargs,
                        mode=mode,
                        response=response,
                        error=e,
                    )
                    raise e
    except RetryError as e:
        hooks.emit_completion_last_attempt(e)
        raise InstructorRetryException(
            e.last_attempt._exception,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            messages=kwargs.get(
                "messages", kwargs.get("contents", kwargs.get("chat_history", []))
            ),
            total_usage=total_usage,
        ) from e


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: type[T] | None,
    context: dict[str, Any] | None,
    args: Any,
    kwargs: Any,
    max_retries: int | AsyncRetrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
    hooks: Hooks | None = None,
) -> T:
    hooks = hooks or Hooks()

    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
        from anthropic.types import Usage as AnthropicUsage

        total_usage = AnthropicUsage(input_tokens=0, output_tokens=0)

    # If max_retries is int, then create a AsyncRetrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
        )
    if not isinstance(max_retries, (AsyncRetrying, Retrying)):
        raise ValueError(
            "max_retries must be an `int` or a `tenacity.AsyncRetrying` object"
        )

    try:
        response = None
        async for attempt in max_retries:
            logger.debug(f"Retrying, attempt: {attempt.retry_state.attempt_number}")
            with attempt:
                try:
                    hooks.emit_completion_arguments(*args, **kwargs)
                    try:
                        response: ChatCompletion = await func(*args, **kwargs)
                    except Exception as e:
                        hooks.emit_completion_error(e)
                        raise e
                    hooks.emit_completion_response(response)
                    stream = kwargs.get("stream", False)
                    response = update_total_usage(response, total_usage)
                    return await process_response_async(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError, AsyncValidationError) as e:
                    hooks.emit_parse_error(e)
                    kwargs = handle_reask_kwargs(
                        kwargs=kwargs,
                        mode=mode,
                        response=response,
                        error=e,
                    )
                    raise e
    except RetryError as e:
        hooks.emit_completion_last_attempt(e)
        raise InstructorRetryException(
            e.last_attempt._exception,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            messages=kwargs.get(
                "messages", kwargs.get("contents", kwargs.get("chat_history", []))
            ),
            total_usage=total_usage,
        ) from e

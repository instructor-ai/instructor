from __future__ import annotations
from functools import wraps
from typing import Any, Callable, Protocol, TypeVar, overload
from collections.abc import Awaitable
import logging

from openai import AsyncOpenAI, OpenAI  # type: ignore[import-not-found]
from pydantic import BaseModel  # type: ignore[import-not-found]
from tenacity import AsyncRetrying, Retrying  # type: ignore[import-not-found]
from typing_extensions import ParamSpec

from ..mode import Mode
from ..utils import is_async
from .create_pipeline import (
    CreatePipelineState,
    cache_lookup_middleware,
    cache_store_middleware,
    context_middleware,
    response_model_middleware,
    retry_async_middleware,
    retry_sync_middleware,
    run_async_pipeline,
    run_sync_pipeline,
    templating_middleware,
)
from .hooks import Hooks

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")


class InstructorChatCompletionCreate(Protocol):
    def __call__(
        self,
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model: ...


class AsyncInstructorChatCompletionCreate(Protocol):
    async def __call__(
        self,
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model: ...
@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.TOOLS,
) -> OpenAI: ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
) -> AsyncOpenAI: ...


@overload
def patch(
    create: Callable[T_ParamSpec, T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate: ...


@overload
def patch(
    create: Awaitable[T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate: ...


def patch(  # type: ignore
    client: OpenAI | AsyncOpenAI | None = None,
    create: Callable[T_ParamSpec, T_Retval] | None = None,
    mode: Mode = Mode.TOOLS,
) -> OpenAI | AsyncOpenAI:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    - `hooks` parameter to hook into the completion process
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is not None:
        func = create
    elif client is not None:
        func = client.chat.completions.create
    else:
        raise ValueError("Either client or create must be provided")

    func_is_async = is_async(func)

    @wraps(func)  # type: ignore
    async def new_create_async(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        from ..cache import BaseCache

        cache: BaseCache | None = kwargs.pop("cache", None)  # type: ignore[assignment]
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl: int | None = (
            cache_ttl_raw if isinstance(cache_ttl_raw, int) else None
        )

        state = CreatePipelineState(
            func=func,  # type: ignore[arg-type]
            mode=mode,
            args=args,
            kwargs=dict(kwargs),
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            strict=strict,
            hooks=hooks,
            cache=cache,
            cache_ttl=cache_ttl,
        )

        middlewares = (
            context_middleware,
            response_model_middleware,
            templating_middleware,
            cache_lookup_middleware,
            retry_async_middleware,
            cache_store_middleware,
        )

        result = await run_async_pipeline(state, middlewares)
        return result  # type: ignore[return-value]

    @wraps(func)  # type: ignore
    def new_create_sync(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        from ..cache import BaseCache

        cache: BaseCache | None = kwargs.pop("cache", None)  # type: ignore[assignment]
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl: int | None = (
            cache_ttl_raw if isinstance(cache_ttl_raw, int) else None
        )

        state = CreatePipelineState(
            func=func,  # type: ignore[arg-type]
            mode=mode,
            args=args,
            kwargs=dict(kwargs),
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            strict=strict,
            hooks=hooks,
            cache=cache,
            cache_ttl=cache_ttl,
        )

        middlewares = (
            context_middleware,
            response_model_middleware,
            templating_middleware,
            cache_lookup_middleware,
            retry_sync_middleware,
            cache_store_middleware,
        )

        result = run_sync_pipeline(state, middlewares)
        return result  # type: ignore[return-value]

    new_create = new_create_async if func_is_async else new_create_sync

    if client is not None:
        client.chat.completions.create = new_create  # type: ignore
        return client
    else:
        return new_create  # type: ignore


def apatch(client: AsyncOpenAI, mode: Mode = Mode.TOOLS) -> AsyncOpenAI:
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    import warnings

    warnings.warn(
        "apatch is deprecated, use patch instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return patch(client, mode=mode)

from __future__ import annotations
from functools import wraps
from typing import (
    Any,
    Callable,
    Protocol,
    TypeVar,
    overload,
)
from collections.abc import Awaitable
from typing_extensions import ParamSpec

from openai import AsyncOpenAI, OpenAI  # type: ignore[import-not-found]
from pydantic import BaseModel  # type: ignore[import-not-found]

from ..processing.response import handle_response_model
from .retry import retry_async, retry_sync
from ..utils import is_async
from .hooks import Hooks
from ..templating import handle_templating

from ..mode import Mode
import logging

from tenacity import (  # type: ignore[import-not-found]
    AsyncRetrying,
    Retrying,
)

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


def handle_context(
    context: dict[str, Any] | None = None,
    validation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Handle the context and validation_context parameters.
    If both are provided, raise an error.
    If validation_context is provided, issue a deprecation warning and use it as context.
    If neither is provided, return None.
    """
    if context is not None and validation_context is not None:
        from .exceptions import ConfigurationError

        raise ConfigurationError(
            "Cannot provide both 'context' and 'validation_context'. Use 'context' instead."
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
        # -----------------------------
        # Cache handling (async path)
        # -----------------------------
        from ..cache import BaseCache, make_cache_key, load_cached_response

        cache: BaseCache | None = kwargs.pop("cache", None)  # type: ignore[assignment]
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl: int | None = (
            cache_ttl_raw if isinstance(cache_ttl_raw, int) else None
        )

        context = handle_context(context, validation_context)

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )  # type: ignore
        new_kwargs = handle_templating(new_kwargs, mode=mode, context=context)

        # Attempt cache lookup **before** hitting retry layer
        if cache is not None and response_model is not None:
            key = make_cache_key(
                messages=new_kwargs.get("messages")
                or new_kwargs.get("contents")
                or new_kwargs.get("chat_history"),
                model=new_kwargs.get("model"),
                response_model=response_model,
                mode=mode.value if hasattr(mode, "value") else str(mode),
            )
            obj = load_cached_response(cache, key, response_model)
            if obj is not None:
                return obj  # type: ignore[return-value]

        response = await retry_async(
            func=func,  # type:ignore
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            strict=strict,
            mode=mode,
            hooks=hooks,
        )

        # Store in cache *after* successful call
        if cache is not None and response_model is not None:
            try:
                from pydantic import BaseModel as _BM  # type: ignore[import-not-found]

                if isinstance(response, _BM):
                    # mypy: ignore-next-line
                    from ..cache import store_cached_response

                    store_cached_response(cache, key, response, ttl=cache_ttl)
            except ModuleNotFoundError:
                pass
        return response  # type: ignore

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
        # -----------------------------
        # Cache handling (sync path)
        # -----------------------------
        from ..cache import BaseCache, make_cache_key, load_cached_response

        cache: BaseCache | None = kwargs.pop("cache", None)  # type: ignore[assignment]
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl: int | None = (
            cache_ttl_raw if isinstance(cache_ttl_raw, int) else None
        )

        context = handle_context(context, validation_context)
        # print(f"instructor.patch: patched_function {func.__name__}")
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )  # type: ignore

        new_kwargs = handle_templating(new_kwargs, mode=mode, context=context)

        # Attempt cache lookup
        if cache is not None and response_model is not None:
            key = make_cache_key(
                messages=new_kwargs.get("messages")
                or new_kwargs.get("contents")
                or new_kwargs.get("chat_history"),
                model=new_kwargs.get("model"),
                response_model=response_model,
                mode=mode.value if hasattr(mode, "value") else str(mode),
            )
            obj = load_cached_response(cache, key, response_model)
            if obj is not None:
                return obj  # type: ignore[return-value]

        response = retry_sync(
            func=func,  # type: ignore
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            hooks=hooks,
            strict=strict,
            kwargs=new_kwargs,
            mode=mode,
        )

        # Save to cache
        if cache is not None and response_model is not None:
            try:
                from pydantic import BaseModel as _BM  # type: ignore[import-not-found]

                if isinstance(response, _BM):
                    # mypy: ignore-next-line
                    from ..cache import store_cached_response

                    store_cached_response(cache, key, response, ttl=cache_ttl)
            except ModuleNotFoundError:
                pass
        return response  # type: ignore

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

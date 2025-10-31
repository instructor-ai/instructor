from __future__ import annotations

"""Pipeline helpers used to orchestrate Instructor's patched completions."""

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import inspect

from tenacity import AsyncRetrying, Retrying  # type: ignore[import-not-found]

from .exceptions import ConfigurationError
from .hooks import Hooks
from .retry import retry_async, retry_sync
from ..cache import (  # type: ignore[import-not-found]
    BaseCache,
    load_cached_response,
    make_cache_key,
    store_cached_response,
)
from ..mode import Mode
from ..processing.response import handle_response_model
from ..templating import handle_templating


@dataclass(slots=True)
class CreatePipelineState:
    """Mutable state that flows through the patched create pipeline."""

    func: Callable[..., Any]
    mode: Mode
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    response_model: type[Any] | None
    validation_context: dict[str, Any] | None
    context: dict[str, Any] | None
    max_retries: int | Retrying | AsyncRetrying
    strict: bool
    hooks: Hooks | None
    cache: BaseCache | None = None
    cache_ttl: int | None = None
    cache_key: str | None = None
    result: Any | None = None
    short_circuit: bool = False


def handle_context(
    context: dict[str, Any] | None = None,
    validation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve backwards compatible validation context arguments."""

    if context is not None and validation_context is not None:
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


def context_middleware(state: CreatePipelineState) -> None:
    """Normalize context arguments for downstream middleware."""

    state.context = handle_context(state.context, state.validation_context)
    state.validation_context = None


def response_model_middleware(state: CreatePipelineState) -> None:
    """Prepare the response model and provider-specific kwargs."""

    response_model, new_kwargs = handle_response_model(
        response_model=state.response_model,
        mode=state.mode,
        **state.kwargs,
    )
    state.response_model = response_model
    state.kwargs = new_kwargs


def templating_middleware(state: CreatePipelineState) -> None:
    """Apply templating on provider payloads using the resolved context."""

    state.kwargs = handle_templating(state.kwargs, mode=state.mode, context=state.context)


def cache_lookup_middleware(state: CreatePipelineState) -> None:
    """Attempt to satisfy the request from cache before hitting the API."""

    if state.cache is None or state.response_model is None:
        return

    messages = (
        state.kwargs.get("messages")
        or state.kwargs.get("contents")
        or state.kwargs.get("chat_history")
    )
    key = make_cache_key(
        messages=messages,
        model=state.kwargs.get("model"),
        response_model=state.response_model,
        mode=state.mode.value if hasattr(state.mode, "value") else str(state.mode),
    )
    state.cache_key = key
    cached = load_cached_response(state.cache, key, state.response_model)
    if cached is not None:
        state.result = cached
        state.short_circuit = True


def cache_store_middleware(state: CreatePipelineState) -> None:
    """Persist successful responses back into the configured cache."""

    if (
        state.cache is None
        or state.response_model is None
        or state.cache_key is None
        or state.result is None
    ):
        return

    try:
        from pydantic import BaseModel  # type: ignore[import-not-found]

        if isinstance(state.result, BaseModel):
            store_cached_response(state.cache, state.cache_key, state.result, ttl=state.cache_ttl)
    except ModuleNotFoundError:  # pragma: no cover - only triggered without pydantic
        return


def retry_sync_middleware(state: CreatePipelineState) -> None:
    """Execute the underlying call with retry logic (synchronous path)."""

    state.result = retry_sync(
        func=state.func,
        response_model=state.response_model,
        args=state.args,
        kwargs=state.kwargs,
        context=state.context,
        max_retries=state.max_retries,  # type: ignore[arg-type]
        strict=state.strict,
        mode=state.mode,
        hooks=state.hooks,
    )


async def retry_async_middleware(state: CreatePipelineState) -> None:
    """Execute the underlying call with retry logic (asynchronous path)."""

    state.result = await retry_async(
        func=state.func,
        response_model=state.response_model,
        args=state.args,
        kwargs=state.kwargs,
        context=state.context,
        max_retries=state.max_retries,  # type: ignore[arg-type]
        strict=state.strict,
        mode=state.mode,
        hooks=state.hooks,
    )


def run_sync_pipeline(
    state: CreatePipelineState,
    middlewares: Sequence[Callable[[CreatePipelineState], None]],
) -> Any:
    """Execute middlewares sequentially for synchronous clients."""

    for middleware in middlewares:
        middleware(state)
        if state.short_circuit:
            break
    return state.result


async def run_async_pipeline(
    state: CreatePipelineState,
    middlewares: Sequence[Callable[[CreatePipelineState], Any]],
) -> Any:
    """Execute middlewares sequentially for asynchronous clients."""

    for middleware in middlewares:
        maybe_result = middleware(state)
        if inspect.isawaitable(maybe_result):
            await maybe_result
        if state.short_circuit:
            break
    return state.result


__all__ = [
    "CreatePipelineState",
    "cache_lookup_middleware",
    "cache_store_middleware",
    "context_middleware",
    "handle_context",
    "response_model_middleware",
    "retry_async_middleware",
    "retry_sync_middleware",
    "run_async_pipeline",
    "run_sync_pipeline",
    "templating_middleware",
]

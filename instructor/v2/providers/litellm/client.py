"""v2 LiteLLM client factory."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Literal, TypeVar, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2


T_Retval = TypeVar("T_Retval")


@overload
def from_litellm(
    completion: Callable[..., Coroutine[Any, Any, T_Retval]],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_litellm(
    completion: Callable[..., Awaitable[T_Retval]],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_litellm(
    completion: Callable[..., object],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


def from_litellm(
    completion: Callable[..., object] | Callable[..., Awaitable[Any]],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: bool | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor client from a LiteLLM completion function."""
    create = patch_v2(func=completion, provider=Provider.OPENAI, mode=mode)
    if async_client is None:
        async_client = inspect.iscoroutinefunction(completion)
    client_type = AsyncInstructor if async_client else Instructor
    return client_type(
        client=None,
        create=create,
        mode=mode,
        provider=Provider.OPENAI,
        **kwargs,
    )


def build_from_model(
    *,
    provider: Provider,  # noqa: ARG001
    model_name: str,  # noqa: ARG001
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,  # noqa: ARG001
    kwargs: dict[str, Any],
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.errors import ConfigurationError

    try:
        from litellm import acompletion, completion
    except ImportError:
        raise ConfigurationError(
            "The litellm package is required to use the LiteLLM provider. "
            "Install it with `pip install litellm`."
        ) from None
    return from_litellm(
        acompletion if async_client else completion,
        mode=mode or Mode.TOOLS,
        async_client=async_client,
        **kwargs,
    )

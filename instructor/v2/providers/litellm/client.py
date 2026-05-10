"""v2 LiteLLM client factory."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2


@overload
def from_litellm(
    completion: Callable[..., Awaitable[Any]],
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_litellm(
    completion: Callable[..., Any],
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


def from_litellm(
    completion: Callable[..., Any] | Callable[..., Awaitable[Any]],
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor client from a LiteLLM completion function."""
    create = patch_v2(func=completion, provider=Provider.OPENAI, mode=mode)
    client_type = AsyncInstructor if inspect.iscoroutinefunction(completion) else Instructor
    return client_type(
        client=None,
        create=create,
        mode=mode,
        provider=Provider.OPENAI,
        **kwargs,
    )

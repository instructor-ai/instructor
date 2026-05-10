from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from tenacity import AsyncRetrying, Retrying

from instructor.mode import Mode
from instructor.utils.core import extract_messages  # noqa: F401
from instructor.utils.providers import Provider
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2

if True:  # typing-only block to avoid runtime dependency on hooks
    from instructor.core.hooks import Hooks

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model")
T_Retval = TypeVar("T_Retval")


def retry_sync(
    func: Callable[..., T_Retval],
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
    """Compatibility wrapper for v2 retry logic (sync)."""
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
    """Compatibility wrapper for v2 retry logic (async)."""
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

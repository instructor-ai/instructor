from __future__ import annotations

import asyncio
import inspect
import warnings
from typing import Any


def close_idle_event_loop() -> None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"There is no current event loop",
                category=DeprecationWarning,
            )
            loop = asyncio.get_event_loop()
    except (RuntimeError, DeprecationWarning):
        return

    if loop.is_running():
        return
    if not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)


def close_provider_client(client: Any, *, async_client: bool = False) -> None:
    if client is None:
        return

    if not async_client and type(client).__module__.startswith("fireworks.client"):
        client._client_v1.close()
        client._image_client_v1.close()
        return

    methods = (
        (("aclose", ()), ("close", ()), ("__aexit__", (None, None, None)))
        if async_client
        else (("close", ()), ("__exit__", (None, None, None)))
    )
    for name, args in methods:
        method = getattr(client, name, None)
        if not callable(method):
            continue
        result = method(*args)
        if inspect.isawaitable(result):

            async def wait_for_close(awaitable: Any) -> None:
                await awaitable

            asyncio.run(wait_for_close(result))
        return

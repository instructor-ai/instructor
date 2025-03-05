from __future__ import annotations

from typing import Any, overload

import instructor
from instructor.client import AsyncInstructor, Instructor


from fireworks.client import Fireworks, AsyncFireworks  # type:ignore


@overload
def from_fireworks(
    client: Fireworks,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    **kwargs: Any,
) -> Instructor:
    ...


@overload
def from_fireworks(
    client: AsyncFireworks,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    **kwargs: Any,
) -> AsyncInstructor:
    ...


def from_fireworks(
    client: Fireworks | AsyncFireworks,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    assert (
        mode
        in {
            instructor.Mode.FIREWORKS_TOOLS,
            instructor.Mode.FIREWORKS_JSON,
        }
    ), "Mode must be one of {instructor.Mode.FIREWORKS_TOOLS, instructor.Mode.FIREWORKS_JSON}"

    assert isinstance(
        client, (AsyncFireworks, Fireworks)
    ), "Client must be an instance of Fireworks or AsyncFireworks"

    if isinstance(client, AsyncFireworks):

        async def async_wrapper(*args: Any, **kwargs: Any):  # type:ignore
            if "stream" in kwargs and kwargs["stream"] is True:
                return client.chat.completions.acreate(*args, **kwargs)  # type:ignore
            return await client.chat.completions.acreate(*args, **kwargs)  # type:ignore

        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.FIREWORKS,
            mode=mode,
            **kwargs,
        )

    if isinstance(client, Fireworks):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),  # type: ignore
            provider=instructor.Provider.FIREWORKS,
            mode=mode,
            **kwargs,
        )

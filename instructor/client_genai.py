# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload

from google.genai import Client

import instructor


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert mode in {
        instructor.Mode.GENAI_TOOLS,
        instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    }, (
        "Mode must be one of {instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS}"
    )

    assert isinstance(
        client,
        (Client),
    ), "Client must be an instance of google.genai.Client"

    if use_async:

        async def async_wrapper(*args: Any, **kwargs: Any):  # type:ignore
            if kwargs.pop("stream", False):
                return await client.aio.models.generate_content_stream(*args, **kwargs)  # type:ignore
            return await client.aio.models.generate_content(*args, **kwargs)  # type:ignore

        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.GENAI,
            mode=mode,
            **kwargs,
        )

    def sync_wrapper(*args: Any, **kwargs: Any):  # type:ignore
        if kwargs.pop("stream", False):
            return client.models.generate_content_stream(*args, **kwargs)  # type:ignore

        return client.models.generate_content(*args, **kwargs)  # type:ignore

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=sync_wrapper, mode=mode),
        provider=instructor.Provider.GENAI,
        mode=mode,
        **kwargs,
    )

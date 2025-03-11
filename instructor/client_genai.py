# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload

from google.genai import Client

import instructor


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
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
    }, "Mode must be one of {instructor.Mode.GENAI_TOOLS}"

    assert isinstance(
        client,
        (Client),
    ), "Client must be an instance of google.genai.Client"

    if use_async:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(
                create=client.aio.models.generate_content, mode=mode
            ),
            provider=instructor.Provider.GENAI,
            mode=mode,
            **kwargs,
        )

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=client.models.generate_content, mode=mode),
        provider=instructor.Provider.GENAI,
        mode=mode,
        **kwargs,
    )

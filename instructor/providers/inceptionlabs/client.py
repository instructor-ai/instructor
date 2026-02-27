from __future__ import annotations

import openai
import instructor
from typing import overload, Any


@overload
def from_inception(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.INCEPTION_JSON,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_inception(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.INCEPTION_JSON,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_inception(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.INCEPTION_JSON,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor client from an Inception Labs client.

    Args:
        client: An Inception Labs client (sync or async)
        mode: The mode to use (INCEPTION_JSON or INCEPTION_TOOLS)
        **kwargs: Additional arguments to pass to the client

    Returns:
        An Instructor client
    """
    valid_modes = {
        instructor.Mode.INCEPTION_JSON,
        instructor.Mode.INCEPTION_TOOLS,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Inception Labs",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of openai.OpenAI or openai.AsyncOpenAI. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, openai.AsyncOpenAI):
        create = client.chat.completions.create
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.INCEPTION,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.INCEPTION,
        mode=mode,
        **kwargs,
    )

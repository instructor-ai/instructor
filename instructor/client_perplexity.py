from __future__ import annotations

import openai
import instructor
from typing import overload, Any


@overload
def from_perplexity(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.PERPLEXITY_JSON,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_perplexity(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.PERPLEXITY_JSON,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_perplexity(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.PERPLEXITY_JSON,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor client from a Perplexity client.

    Args:
        client: A Perplexity client (sync or async)
        mode: The mode to use for the client (must be PERPLEXITY_JSON)
        **kwargs: Additional arguments to pass to the client

    Returns:
        An Instructor client
    """
    assert mode == instructor.Mode.PERPLEXITY_JSON, "Mode must be PERPLEXITY_JSON"

    assert isinstance(
        client,
        (openai.OpenAI, openai.AsyncOpenAI),
    ), "Client must be an instance of openai.OpenAI or openai.AsyncOpenAI"

    if isinstance(client, openai.AsyncOpenAI):
        create = client.chat.completions.create
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.PERPLEXITY,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.PERPLEXITY,
        mode=mode,
        **kwargs,
    )

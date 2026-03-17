from __future__ import annotations

import openai
import instructor
from typing import overload, Any


@overload
def from_minimax(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_minimax(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_minimax(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor client from a MiniMax-configured OpenAI client.

    MiniMax provides an OpenAI-compatible API, so this uses the standard
    OpenAI SDK with MiniMax's base URL and API key.

    Args:
        client: An OpenAI client configured with MiniMax's base URL
        mode: The mode to use (MINIMAX_TOOLS or MINIMAX_JSON)
        **kwargs: Additional arguments to pass to the client

    Returns:
        An Instructor client

    Example:
        >>> import openai
        >>> import instructor
        >>> client = instructor.from_minimax(
        ...     openai.OpenAI(
        ...         api_key="your-minimax-api-key",
        ...         base_url="https://api.minimax.io/v1",
        ...     ),
        ...     mode=instructor.Mode.MINIMAX_TOOLS,
        ... )
    """
    valid_modes = {
        instructor.Mode.MINIMAX_TOOLS,
        instructor.Mode.MINIMAX_JSON,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="MiniMax",
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
            provider=instructor.Provider.MINIMAX,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.MINIMAX,
        mode=mode,
        **kwargs,
    )

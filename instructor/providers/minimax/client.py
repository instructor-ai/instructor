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
    """Create an Instructor client from a MiniMax OpenAI-compatible client.

    Args:
        client: An ``openai.OpenAI`` or ``openai.AsyncOpenAI`` instance pointed at
            the MiniMax API (``base_url="https://api.minimax.io/v1"``).
        mode: The structured-output mode to use.  Defaults to
            ``MINIMAX_TOOLS`` (tool calling).  Use ``MINIMAX_JSON`` for
            system-prompt-based JSON output when tool calling is not desired.
        **kwargs: Additional arguments forwarded to the ``Instructor`` constructor.

    Returns:
        An ``Instructor`` or ``AsyncInstructor`` instance.
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

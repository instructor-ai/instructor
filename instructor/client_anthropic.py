from __future__ import annotations

import anthropic
import instructor

from typing import overload, Any


@overload
def from_anthropic(
    client: (
        anthropic.Anthropic | anthropic.AnthropicBedrock | anthropic.AnthropicVertex
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    beta: bool = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_anthropic(
    client: (
        anthropic.AsyncAnthropic
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    beta: bool = False,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_anthropic(
    client: (
        anthropic.Anthropic
        | anthropic.AsyncAnthropic
        | anthropic.AnthropicBedrock
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
        | anthropic.AnthropicVertex
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    beta: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor instance from an Anthropic client.

    Args:
        client: An instance of Anthropic client (sync or async)
        mode: The mode to use for the client (ANTHROPIC_JSON or ANTHROPIC_TOOLS)
        beta: Whether to use beta API features (uses client.beta.messages.create)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not one of the valid Anthropic modes
        ClientError: If client is not a valid Anthropic client instance
    """
    valid_modes = {
        instructor.Mode.ANTHROPIC_JSON,
        instructor.Mode.ANTHROPIC_TOOLS,
        instructor.Mode.ANTHROPIC_REASONING_TOOLS,
        instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Anthropic",
            valid_modes=[str(m) for m in valid_modes],
        )

    valid_client_types = (
        anthropic.Anthropic,
        anthropic.AsyncAnthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
        anthropic.AsyncAnthropicVertex,
    )

    if not isinstance(client, valid_client_types):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    if beta:
        create = client.beta.messages.create
    else:
        create = client.messages.create

    if isinstance(
        client,
        (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex),
    ):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )

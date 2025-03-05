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
        enable_prompt_caching: Whether to enable prompt caching (requires Anthropic or AsyncAnthropic client)
        beta: Whether to use beta API features (uses client.beta.messages.create)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        TypeError: If enable_prompt_caching is True and client is not Anthropic or AsyncAnthropic
        AssertionError: If mode is not ANTHROPIC_JSON or ANTHROPIC_TOOLS
    """
    assert mode in {
        instructor.Mode.ANTHROPIC_JSON,
        instructor.Mode.ANTHROPIC_TOOLS,
        instructor.Mode.ANTHROPIC_REASONING_TOOLS,
    }, (
        "Mode be one of {instructor.Mode.ANTHROPIC_JSON, instructor.Mode.ANTHROPIC_TOOLS, instructor.Mode.ANTHROPIC_REASONING_TOOLS}"
    )

    assert isinstance(
        client,
        (
            anthropic.Anthropic,
            anthropic.AsyncAnthropic,
            anthropic.AnthropicBedrock,
            anthropic.AnthropicVertex,
            anthropic.AsyncAnthropicBedrock,
            anthropic.AsyncAnthropicVertex,
        ),
    ), (
        "Client must be an instance of {anthropic.Anthropic, anthropic.AsyncAnthropic, anthropic.AnthropicBedrock, anthropic.AsyncAnthropicBedrock,  anthropic.AnthropicVertex, anthropic.AsyncAnthropicVertex}"
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

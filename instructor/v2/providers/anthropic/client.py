"""v2 Anthropic client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

from typing import Any, overload

import anthropic

from instructor import AsyncInstructor, Instructor, Mode, Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.anthropic import handlers  # noqa: F401


@overload
def from_anthropic(
    client: (
        anthropic.Anthropic | anthropic.AnthropicBedrock | anthropic.AnthropicVertex
    ),
    mode: Mode = Mode.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_anthropic(
    client: (
        anthropic.AsyncAnthropic
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
    ),
    mode: Mode = Mode.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_anthropic(
    client: (
        anthropic.Anthropic
        | anthropic.AsyncAnthropic
        | anthropic.AnthropicBedrock
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
        | anthropic.AnthropicVertex
    ),
    mode: Mode = Mode.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an Anthropic client using v2 registry.

    Args:
        client: An instance of Anthropic client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        beta: Whether to use beta API features (uses client.beta.messages.create)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ValueError: If mode is not registered
        TypeError: If client is not a valid Anthropic client instance

    Examples:
        >>> import anthropic
        >>> from instructor import Mode
        >>> from instructor.v2.providers.anthropic import from_anthropic
        >>>
        >>> client = anthropic.Anthropic()
        >>> instructor_client = from_anthropic(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use JSON mode
        >>> instructor_client = from_anthropic(client, mode=Mode.JSON)
    """
    from instructor.v2.core.registry import mode_registry

    # Validate mode is registered
    if not mode_registry.is_registered(Provider.ANTHROPIC, mode):
        from instructor.core.exceptions import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)
        raise ModeError(
            mode=mode.value,
            provider=Provider.ANTHROPIC.value,
            valid_modes=[m.value for m in available_modes],
        )

    # Validate client type
    valid_client_types = (
        anthropic.Anthropic,
        anthropic.AsyncAnthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
        anthropic.AsyncAnthropicVertex,
    )

    if not isinstance(client, valid_client_types):
        from instructor.core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function (beta or regular)
    if beta:
        create = client.beta.messages.create
    else:
        create = client.messages.create

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.ANTHROPIC,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(
        client,
        (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex),
    ):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )

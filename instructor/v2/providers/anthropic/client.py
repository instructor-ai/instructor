"""v2 Anthropic client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

from typing import Any, overload

import anthropic

from instructor import AsyncInstructor, Instructor
from instructor.v2.core import ModeType, Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.anthropic import handlers  # noqa: F401


@overload
def from_anthropic(
    client: (
        anthropic.Anthropic | anthropic.AnthropicBedrock | anthropic.AnthropicVertex
    ),
    mode_type: ModeType = ModeType.TOOLS,
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
    mode_type: ModeType = ModeType.TOOLS,
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
    mode_type: ModeType = ModeType.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an Anthropic client using v2 registry.

    v2 uses hierarchical (Provider, ModeType) design instead of flat Mode enum:
        - from_anthropic(client, ModeType.TOOLS)  # v2
        - from_anthropic(client, Mode.ANTHROPIC_TOOLS)  # v1

    Args:
        client: An instance of Anthropic client (sync or async)
        mode_type: The mode type to use (TOOLS or JSON)
        beta: Whether to use beta API features (uses client.beta.messages.create)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ValueError: If mode_type is not registered
        TypeError: If client is not a valid Anthropic client instance

    Examples:
        >>> import anthropic
        >>> from instructor.v2 import from_anthropic, ModeType
        >>>
        >>> client = anthropic.Anthropic()
        >>> instructor_client = from_anthropic(client, ModeType.TOOLS)
        >>>
        >>> # Or use JSON mode
        >>> instructor_client = from_anthropic(client, ModeType.JSON)
    """
    from instructor.v2.core.registry import mode_registry

    # Validate mode_type is registered
    if not mode_registry.is_registered(Provider.ANTHROPIC, mode_type):
        available_modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)
        raise ValueError(
            f"ModeType.{mode_type.name} is not registered for Anthropic. "
            f"Available modes: {[f'ModeType.{m.name}' for m in available_modes]}"
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
        raise TypeError(
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
        mode_type=mode_type,
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
            mode=(Provider.ANTHROPIC, mode_type),  # v2 uses tuple
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.ANTHROPIC,
            mode=(Provider.ANTHROPIC, mode_type),  # v2 uses tuple
            **kwargs,
        )

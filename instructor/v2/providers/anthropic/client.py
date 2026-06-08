"""v2 Anthropic client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.errors import ClientError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.anthropic import handlers  # noqa: F401

if TYPE_CHECKING:
    import anthropic
else:
    try:
        import anthropic
    except ImportError:
        anthropic = None


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
    if anthropic is None:
        raise ClientError(
            "anthropic is not installed. Install it with: pip install anthropic"
        )

    create_path = "beta.messages.create" if beta else None
    return create_instructor(
        client,
        provider=Provider.ANTHROPIC,
        mode=mode,
        model=model,
        create_path=create_path,
        async_create_path=create_path,
        sync_types=(
            anthropic.Anthropic,
            anthropic.AnthropicBedrock,
            anthropic.AnthropicVertex,
        ),
        async_types=(
            anthropic.AsyncAnthropic,
            anthropic.AsyncAnthropicBedrock,
            anthropic.AsyncAnthropicVertex,
        ),
        **kwargs,
    )


def build_from_model(
    *,
    provider: Provider,  # noqa: ARG001
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
) -> Instructor | AsyncInstructor:
    from instructor import __version__
    from instructor.v2.core.errors import ConfigurationError

    if anthropic is None:
        raise ConfigurationError(
            "The anthropic package is required to use the Anthropic provider. "
            "Install it with `pip install anthropic`."
        )
    factory = anthropic.AsyncAnthropic if async_client else anthropic.Anthropic
    client = factory(
        api_key=api_key,
        default_headers={"User-Agent": f"instructor/{__version__}"},
    )
    kwargs.setdefault("max_tokens", 4096)
    return from_anthropic(client, model=model_name, mode=mode or Mode.TOOLS, **kwargs)

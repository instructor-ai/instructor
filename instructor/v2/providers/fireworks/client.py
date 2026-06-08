"""v2 Fireworks client factory.

Creates Instructor instances for Fireworks using v2 hierarchical registry system.
Fireworks uses an OpenAI-compatible API, so the client factory follows the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
# Fireworks uses OpenAI-compatible API, so handlers are registered via OpenAI handlers
from instructor.v2.providers.openai import handlers  # noqa: F401

if TYPE_CHECKING:
    from fireworks.client import AsyncFireworks, Fireworks
else:
    try:
        from fireworks.client import AsyncFireworks, Fireworks
    except ImportError:
        AsyncFireworks = None
        Fireworks = None


@overload
def from_fireworks(
    client: Fireworks,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_fireworks(
    client: AsyncFireworks,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_fireworks(
    client: Fireworks | AsyncFireworks,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Fireworks client using v2 registry.

    Fireworks uses an OpenAI-compatible API, so this factory follows the same pattern
    as the OpenAI factory. Fireworks supports TOOLS and MD_JSON modes.

    Args:
        client: An instance of Fireworks client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Fireworks
        ClientError: If client is not a valid Fireworks client instance or fireworks not installed

    Examples:
        >>> from fireworks.client import Fireworks
        >>> from instructor import Mode
        >>> from instructor.v2.providers.fireworks import from_fireworks
        >>>
        >>> client = Fireworks()
        >>> instructor_client = from_fireworks(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_fireworks(client, mode=Mode.MD_JSON)
    """
    if Fireworks is None or AsyncFireworks is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "fireworks is not installed. Install it with: pip install fireworks-ai"
        )

    return create_instructor(
        client,
        provider=Provider.FIREWORKS,
        mode=mode,
        model=model,
        sync_types=(Fireworks,),
        async_types=(AsyncFireworks,),
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
    """Construct the native Fireworks client for `from_provider`."""
    if Fireworks is None or AsyncFireworks is None:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The fireworks-ai package is required to use the Fireworks provider. "
            "Install it with `pip install fireworks-ai`."
        )
    client = (
        AsyncFireworks(api_key=api_key) if async_client else Fireworks(api_key=api_key)
    )
    return from_fireworks(client, model=model_name, mode=mode or Mode.TOOLS, **kwargs)

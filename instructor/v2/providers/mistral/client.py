"""v2 Mistral client factory.

Creates Instructor instances for Mistral AI using v2 hierarchical registry system.

Mistral has a unique API structure:
- Single client class (Mistral) with both sync and async methods
- Uses `chat.complete()` / `chat.complete_async()` for completions
- Uses `chat.stream()` / `chat.stream_async()` for streaming
- The `use_async` parameter determines which methods to use
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.mistral import handlers  # noqa: F401

if TYPE_CHECKING:
    from mistralai import Mistral
else:
    try:
        from mistralai import Mistral
    except ImportError:
        Mistral = None


@overload
def from_mistral(
    client: Mistral,
    mode: Mode = Mode.TOOLS,
    use_async: Literal[False] = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: Mode = Mode.TOOLS,
    use_async: Literal[True] = True,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: Mode = Mode.TOOLS,
    use_async: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor: ...


def from_mistral(
    client: Mistral,
    mode: Mode = Mode.TOOLS,
    use_async: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Mistral client using v2 registry.

    Mistral uses a single client class with both sync and async methods.
    The `use_async` parameter determines which methods to use.

    Args:
        client: An instance of Mistral client
        mode: The mode to use (defaults to Mode.TOOLS)
        use_async: Whether to use async methods (defaults to False)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on use_async)

    Raises:
        ModeError: If mode is not registered for Mistral
        ClientError: If client is not a valid Mistral client instance or mistralai not installed

    Examples:
        >>> from mistralai import Mistral
        >>> from instructor import Mode
        >>> from instructor.v2.providers.mistral import from_mistral
        >>>
        >>> client = Mistral(api_key="...")
        >>> instructor_client = from_mistral(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use async mode
        >>> async_client = from_mistral(client, mode=Mode.TOOLS, use_async=True)
        >>>
        >>> # Or use structured outputs
        >>> instructor_client = from_mistral(client, mode=Mode.JSON_SCHEMA)
    """
    return create_instructor(
        client,
        provider=Provider.MISTRAL,
        mode=mode,
        model=model,
        use_async=use_async,
        sync_types=(Mistral,) if Mistral is not None else None,
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
    """Construct the native Mistral client for `from_provider`."""
    if Mistral is None:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The mistralai package is required to use the Mistral provider. "
            "Install it with `pip install mistralai`."
        )
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is not set. "
            "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
        )
    client = Mistral(api_key=api_key)
    return from_mistral(
        client,
        model=model_name,
        mode=mode or Mode.TOOLS,
        use_async=async_client,
        **kwargs,
    )

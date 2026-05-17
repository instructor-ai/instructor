"""v2 Mistral client factory.

Creates Instructor instances for Mistral AI using v2 hierarchical registry system.

Mistral has a unique API structure:
- Single client class (Mistral) with both sync and async methods
- Uses `chat.complete()` / `chat.complete_async()` for completions
- Uses `chat.stream()` / `chat.stream_async()` for streaming
- The `use_async` parameter determines which methods to use
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

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
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Check if mistralai is installed
    if Mistral is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "mistralai is not installed. Install it with: pip install mistralai"
        )

    # Normalize provider-specific modes to generic modes
    normalized_mode = normalize_mode(Provider.MISTRAL, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.MISTRAL, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.MISTRAL)
        raise ModeError(
            mode=mode.value,
            provider=Provider.MISTRAL.value,
            valid_modes=[m.value for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    if not isinstance(client, Mistral):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of mistralai.Mistral. "
            f"Got: {type(client).__name__}"
        )

    # Create wrapper functions for Mistral's unique API
    if use_async:

        async def async_wrapper(*args: Any, **wrapper_kwargs: Any) -> Any:
            """Async wrapper that handles streaming."""
            if wrapper_kwargs.pop("stream", False):
                return await client.chat.stream_async(*args, **wrapper_kwargs)
            return await client.chat.complete_async(*args, **wrapper_kwargs)

        # Patch using v2 registry
        patched_create = patch_v2(
            func=async_wrapper,
            provider=Provider.MISTRAL,
            mode=mode,
            default_model=model,
        )

        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )
    else:

        def sync_wrapper(*args: Any, **wrapper_kwargs: Any) -> Any:
            """Sync wrapper that handles streaming."""
            if wrapper_kwargs.pop("stream", False):
                return client.chat.stream(*args, **wrapper_kwargs)
            return client.chat.complete(*args, **wrapper_kwargs)

        # Patch using v2 registry
        patched_create = patch_v2(
            func=sync_wrapper,
            provider=Provider.MISTRAL,
            mode=mode,
            default_model=model,
        )

        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )

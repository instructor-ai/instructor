"""v2 Cohere client factory.

Creates Instructor instances using v2 hierarchical registry system.
Supports both Cohere V1 and V2 client APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.cohere import handlers  # noqa: F401

if TYPE_CHECKING:
    import cohere
else:
    try:
        import cohere
    except ImportError:
        cohere = None  # type: ignore[assignment]


@overload
def from_cohere(
    client: cohere.Client,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cohere(
    client: cohere.ClientV2,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClientV2,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient | cohere.ClientV2 | cohere.AsyncClientV2,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Cohere client using v2 registry.

    Args:
        client: A Cohere client instance (V1 or V2, sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Cohere
        ClientError: If client is not a valid Cohere client instance

    Examples:
        >>> import cohere
        >>> from instructor import Mode
        >>> from instructor.v2.providers.cohere import from_cohere
        >>>
        >>> # V2 client (recommended)
        >>> client = cohere.ClientV2()
        >>> instructor_client = from_cohere(client, mode=Mode.TOOLS)
        >>>
        >>> # V1 client
        >>> client = cohere.Client()
        >>> instructor_client = from_cohere(client, mode=Mode.JSON_SCHEMA)
    """
    is_v2 = cohere is not None and isinstance(
        client, (cohere.ClientV2, cohere.AsyncClientV2)
    )
    kwargs["_cohere_client_version"] = "v2" if is_v2 else "v1"
    sync_types = (cohere.Client, cohere.ClientV2) if cohere is not None else None
    async_types = (
        (cohere.AsyncClient, cohere.AsyncClientV2) if cohere is not None else None
    )
    return create_instructor(
        client,
        provider=Provider.COHERE,
        mode=mode,
        sync_types=sync_types,
        async_types=async_types,
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
    """Construct the native Cohere client for `from_provider`."""
    if cohere is None:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The cohere package is required to use the Cohere provider. "
            "Install it with `pip install cohere`."
        )
    client = (
        cohere.AsyncClientV2(api_key=api_key)
        if async_client
        else cohere.ClientV2(api_key=api_key)
    )
    return from_cohere(client, mode=mode or Mode.TOOLS, model=model_name, **kwargs)

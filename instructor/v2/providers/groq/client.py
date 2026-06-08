"""v2 Groq client factory.

Creates Instructor instances for Groq using v2 hierarchical registry system.
Groq uses an OpenAI-compatible API, so the client factory follows the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
# Groq uses OpenAI-compatible API, so handlers are registered via OpenAI handlers
from instructor.v2.providers.openai import handlers  # noqa: F401

if TYPE_CHECKING:
    import groq
else:
    try:
        import groq
    except ImportError:
        groq = None


@overload
def from_groq(
    client: groq.Groq,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_groq(
    client: groq.AsyncGroq,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_groq(
    client: groq.Groq | groq.AsyncGroq,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Groq client using v2 registry.

    Groq uses an OpenAI-compatible API, so this factory follows the same pattern
    as the OpenAI factory. Groq supports TOOLS and MD_JSON modes.

    Args:
        client: An instance of Groq client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Groq
        ClientError: If client is not a valid Groq client instance or groq not installed

    Examples:
        >>> import groq
        >>> from instructor import Mode
        >>> from instructor.v2.providers.groq import from_groq
        >>>
        >>> client = groq.Groq()
        >>> instructor_client = from_groq(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_groq(client, mode=Mode.MD_JSON)
    """
    if groq is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError("groq is not installed. Install it with: pip install groq")

    return create_instructor(
        client,
        provider=Provider.GROQ,
        mode=mode,
        model=model,
        sync_types=(groq.Groq,),
        async_types=(groq.AsyncGroq,),
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
    """Construct the native Groq client for `from_provider`."""
    if groq is None:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The groq package is required to use the Groq provider. "
            "Install it with `pip install groq`."
        )
    client = (
        groq.AsyncGroq(api_key=api_key) if async_client else groq.Groq(api_key=api_key)
    )
    return from_groq(client, model=model_name, mode=mode or Mode.TOOLS, **kwargs)

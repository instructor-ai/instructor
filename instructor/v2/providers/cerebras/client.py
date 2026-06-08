"""v2 Cerebras client factory.

Creates Instructor instances for Cerebras using v2 hierarchical registry system.
Cerebras uses an OpenAI-compatible API, so the client factory follows the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
# Cerebras uses OpenAI-compatible API, so handlers are registered via OpenAI handlers
from instructor.v2.providers.openai import handlers  # noqa: F401

if TYPE_CHECKING:
    from cerebras.cloud.sdk import AsyncCerebras, Cerebras
else:
    try:
        from cerebras.cloud.sdk import AsyncCerebras, Cerebras
    except ImportError:
        AsyncCerebras = None
        Cerebras = None


@overload
def from_cerebras(
    client: Cerebras,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cerebras(
    client: AsyncCerebras,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cerebras(
    client: Cerebras | AsyncCerebras,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Cerebras client using v2 registry.

    Cerebras uses an OpenAI-compatible API, so this factory follows the same pattern
    as the OpenAI factory. Cerebras supports TOOLS and MD_JSON modes.

    Args:
        client: An instance of Cerebras client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Cerebras
        ClientError: If client is not a valid Cerebras client instance or SDK not installed

    Examples:
        >>> from cerebras.cloud.sdk import Cerebras
        >>> from instructor import Mode
        >>> from instructor.v2.providers.cerebras import from_cerebras
        >>>
        >>> client = Cerebras()
        >>> instructor_client = from_cerebras(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_cerebras(client, mode=Mode.MD_JSON)
    """
    if Cerebras is None or AsyncCerebras is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "cerebras is not installed. Install it with: pip install cerebras-cloud-sdk"
        )

    return create_instructor(
        client,
        provider=Provider.CEREBRAS,
        mode=mode,
        model=model,
        sync_types=(Cerebras,),
        async_types=(AsyncCerebras,),
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
    """Construct the native Cerebras client for `from_provider`."""
    if Cerebras is None or AsyncCerebras is None:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The cerebras package is required to use the Cerebras provider. "
            "Install it with `pip install cerebras`."
        )
    client = (
        AsyncCerebras(api_key=api_key) if async_client else Cerebras(api_key=api_key)
    )
    return from_cerebras(client, model=model_name, mode=mode or Mode.TOOLS, **kwargs)

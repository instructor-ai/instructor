"""v2 Writer client factory.

Creates Instructor instances for Writer using v2 hierarchical registry system.
Writer uses the writerai SDK with Writer and AsyncWriter clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.writer import handlers  # noqa: F401

if TYPE_CHECKING:
    from writerai import AsyncWriter, Writer
else:
    try:
        from writerai import AsyncWriter, Writer
    except ImportError:
        AsyncWriter = None
        Writer = None


@overload
def from_writer(
    client: Writer,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_writer(
    client: AsyncWriter,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_writer(
    client: Writer | AsyncWriter,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Writer client using v2 registry.

    Writer uses the writerai SDK and supports TOOLS, JSON_SCHEMA, and MD_JSON modes.
    The API uses `client.chat.chat` for completions.

    Args:
        client: An instance of Writer client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Writer
        ClientError: If client is not a valid Writer client instance or SDK not installed

    Examples:
        >>> from writerai import Writer
        >>> from instructor import Mode
        >>> from instructor.v2.providers.writer import from_writer
        >>>
        >>> client = Writer()
        >>> instructor_client = from_writer(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_writer(client, mode=Mode.MD_JSON)
    """
    if Writer is None or AsyncWriter is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "writerai is not installed. Install it with: pip install writer-sdk"
        )

    return create_instructor(
        client,
        provider=Provider.WRITER,
        mode=mode,
        model=model,
        sync_types=(Writer,),
        async_types=(AsyncWriter,),
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
    """Construct the native Writer client for `from_provider`."""
    if Writer is None or AsyncWriter is None:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The writerai package is required to use the Writer provider. "
            "Install it with `pip install writer-sdk`."
        )
    client = AsyncWriter(api_key=api_key) if async_client else Writer(api_key=api_key)
    return from_writer(client, model=model_name, mode=mode or Mode.TOOLS, **kwargs)

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered (decorators auto-register on import)
from . import handlers  # noqa: F401

if TYPE_CHECKING:
    from google.genai import Client
else:
    try:
        from google.genai import Client
    except ImportError:
        Client = None  # type: ignore[assignment]


@overload
def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """
    Create a v2 Instructor client from a google.genai.Client instance.

    Supports generic modes (TOOLS, JSON).

    Args:
        client: google.genai.Client instance
        mode: Mode to use (defaults to Mode.TOOLS)
        use_async: Whether to use async client
        model: Default model name to inject into requests if not provided
        **kwargs: Additional kwargs passed to Instructor constructor
    """
    return create_instructor(
        client,
        provider=Provider.GENAI,
        mode=mode,
        model=model,
        use_async=use_async,
        sync_types=(Client,) if Client is not None else None,
        **kwargs,
    )


def build_from_model(
    *,
    provider: Provider,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.errors import ConfigurationError

    if Client is None:
        raise ConfigurationError(
            "The google-genai package is required to use the Google provider. "
            "Install it with `pip install google-genai`."
        )
    if provider is Provider.GENERATIVE_AI:
        warnings.warn(
            "The 'generative-ai' provider is deprecated. Use 'google' provider instead. "
            "Example: instructor.from_provider('google/gemini-pro')",
            DeprecationWarning,
            stacklevel=2,
        )
    client_kwargs = {
        key: kwargs.pop(key)
        for key in (
            "debug_config",
            "http_options",
            "credentials",
            "project",
            "location",
        )
        if key in kwargs
    }
    client = Client(
        vertexai=kwargs.pop("vertexai", False),
        api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
        **client_kwargs,
    )
    return from_genai(
        client,
        mode=mode or Mode.TOOLS,
        use_async=async_client,
        model=kwargs.pop("model", model_name),
        **kwargs,
    )

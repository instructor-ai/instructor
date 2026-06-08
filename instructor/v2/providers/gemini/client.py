"""v2 Gemini client factory."""

from __future__ import annotations

import importlib
import os
from typing import Any, Literal, cast, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.client_factory import create_instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider

# Ensure handlers are registered.
from instructor.v2.providers.gemini import handlers  # noqa: F401

try:
    genai: Any = importlib.import_module("google.generativeai")
except ImportError:
    genai = None


@overload
def from_gemini(
    client: Any,
    mode: Mode = Mode.MD_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_gemini(
    client: Any,
    mode: Mode = Mode.MD_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_gemini(
    client: Any,
    mode: Mode = Mode.MD_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    generative_model_type = getattr(genai, "GenerativeModel", None)
    sync_types = (
        (generative_model_type,) if isinstance(generative_model_type, type) else None
    )

    return create_instructor(
        client,
        provider=Provider.GEMINI,
        mode=mode,
        use_async=use_async,
        sync_types=sync_types,
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
    from instructor.v2.core.errors import ConfigurationError

    if genai is None:
        raise ConfigurationError(
            "The google-generativeai package is required to use the Gemini provider. "
            "Install it with `pip install google-generativeai`."
        )
    client_sdk = cast(Any, genai)
    resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if resolved_key:
        client_sdk.configure(api_key=resolved_key)
    return from_gemini(
        client_sdk.GenerativeModel(model_name),
        mode=mode or Mode.MD_JSON,
        use_async=async_client,
        **kwargs,
    )


__all__ = ["build_from_model", "from_gemini"]

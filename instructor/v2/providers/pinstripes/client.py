"""v2 Pinstripes client factory.

Creates Instructor instances for Pinstripes using the v2 hierarchical registry
system. Pinstripes exposes an OpenAI-compatible API so the client factory
delegates to _from_openai_compat.
"""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import _from_openai_compat

# Ensure OpenAI handlers are registered (Pinstripes is OpenAI-compatible).
from instructor.v2.providers.openai import handlers  # noqa: F401


@overload
def from_pinstripes(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_pinstripes(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_pinstripes(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Pinstripes-configured OpenAI client.

    Pinstripes is an OpenAI-compatible LLM inference API.  Pass an
    ``openai.OpenAI`` (or ``openai.AsyncOpenAI``) client pointed at
    ``https://pinstripes.io/v1`` and this factory wraps it with instructor's
    structured-output machinery.

    Args:
        client: An ``openai.OpenAI`` or ``openai.AsyncOpenAI`` instance
            configured with ``base_url="https://pinstripes.io/v1"`` and the
            ``PINSTRIPES_API_KEY`` secret.
        mode: Extraction mode (defaults to ``Mode.TOOLS``).
        model: Optional model name to inject when not supplied per-request.
        **kwargs: Additional keyword arguments forwarded to the Instructor
            constructor.

    Returns:
        An ``Instructor`` or ``AsyncInstructor`` instance.

    Raises:
        ModeError: If *mode* is not supported by Pinstripes.
        ClientError: If *client* is not a valid ``openai.OpenAI``/
            ``openai.AsyncOpenAI`` instance.

    Examples:
        >>> import os
        >>> from openai import OpenAI
        >>> import instructor
        >>>
        >>> client = instructor.from_pinstripes(
        ...     OpenAI(
        ...         api_key=os.environ["PINSTRIPES_API_KEY"],
        ...         base_url="https://pinstripes.io/v1",
        ...     ),
        ...     model="ps/deepseek-v4-flash",
        ... )
    """
    return _from_openai_compat(
        client=client,
        provider=Provider.PINSTRIPES,
        mode=mode,
        model=model,
        **kwargs,
    )


__all__ = ["from_pinstripes"]

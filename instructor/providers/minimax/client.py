from __future__ import annotations

import os
from typing import Any, overload

import openai

import instructor


DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.chat/v1"


def _build_openai_client(
    async_client: bool,
    api_key: str | None,
    base_url: str | None,
) -> openai.OpenAI | openai.AsyncOpenAI:
    """Construct an OpenAI-compatible client pointed at MiniMax."""
    resolved_api_key = api_key or os.environ.get("MINIMAX_API_KEY")
    if not resolved_api_key:
        from ...core.exceptions import ConfigurationError

        raise ConfigurationError(
            "MINIMAX_API_KEY is not set. "
            "Set it with `export MINIMAX_API_KEY=<your-api-key>` or "
            "pass it as a kwarg `api_key=<your-api-key>`."
        )

    resolved_base_url = base_url or DEFAULT_MINIMAX_BASE_URL
    if async_client:
        return openai.AsyncOpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
    return openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)


@overload
def from_minimax(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_minimax(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_minimax(
    client: None = None,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor: ...


def from_minimax(
    client: openai.OpenAI | openai.AsyncOpenAI | None = None,
    mode: instructor.Mode = instructor.Mode.MINIMAX_TOOLS,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor client for the MiniMax API.

    MiniMax exposes an OpenAI-compatible chat completions endpoint, so this
    factory wraps an ``openai`` client configured to talk to MiniMax. If no
    client is provided, one is constructed automatically using the
    ``MINIMAX_API_KEY`` environment variable and the default MiniMax base
    URL (``https://api.minimax.chat/v1``).

    Args:
        client: An optional pre-configured ``openai.OpenAI`` or
            ``openai.AsyncOpenAI`` client pointing at MiniMax.
        mode: The mode to use for the client. Must be one of
            ``MINIMAX_TOOLS`` or ``MINIMAX_JSON``.
        api_key: API key to use when constructing a client automatically.
            Falls back to the ``MINIMAX_API_KEY`` environment variable.
        base_url: Override the default MiniMax base URL.
        async_client: When ``client`` is ``None``, controls whether an async
            or sync client is constructed.
        **kwargs: Additional arguments forwarded to the Instructor client.

    Returns:
        An ``Instructor`` or ``AsyncInstructor`` instance.
    """
    valid_modes = {instructor.Mode.MINIMAX_TOOLS, instructor.Mode.MINIMAX_JSON}

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="MiniMax",
            valid_modes=[str(m) for m in valid_modes],
        )

    if client is None:
        client = _build_openai_client(
            async_client=async_client,
            api_key=api_key,
            base_url=base_url,
        )

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        from ...core.exceptions import ClientError

        raise ClientError(
            "Client must be an instance of openai.OpenAI or openai.AsyncOpenAI. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, openai.AsyncOpenAI):
        create = client.chat.completions.create
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.MINIMAX,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.MINIMAX,
        mode=mode,
        **kwargs,
    )

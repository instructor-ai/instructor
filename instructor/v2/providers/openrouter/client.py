"""v2 OpenRouter client factory."""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import (
    _from_openai_compat,
    compatible_model_builder,
)

# Ensure OpenRouter handlers are registered (overrides JSON_SCHEMA).
from instructor.v2.providers.openrouter import handlers  # noqa: F401


@overload
def from_openrouter(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_openrouter(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_openrouter(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    return _from_openai_compat(
        client=client,
        provider=Provider.OPENROUTER,
        mode=mode,
        model=model,
        **kwargs,
    )


build_from_model = compatible_model_builder(
    from_openrouter,
    env_var="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1",
)


__all__ = ["build_from_model", "from_openrouter"]

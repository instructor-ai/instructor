"""Compatibility exports for provider-owned schema generation helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = [
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
]


def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Compatibility wrapper for the provider-owned OpenAI schema helper."""
    from instructor.v2.providers.openai.schema import (
        generate_openai_schema as _generate_openai_schema,
    )
    return _generate_openai_schema(model)


def generate_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Compatibility wrapper for the provider-owned Anthropic schema helper."""
    from instructor.v2.providers.anthropic.schema import (
        generate_anthropic_schema as _generate_anthropic_schema,
    )
    return _generate_anthropic_schema(model)


def generate_gemini_schema(model: type[BaseModel]) -> Any:
    """Compatibility wrapper for the provider-owned Gemini schema helper."""
    from instructor.v2.providers.gemini.schema import (
        generate_gemini_schema as _generate_gemini_schema,
    )
    return _generate_gemini_schema(model)

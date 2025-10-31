"""Validation models shared across instructor."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ..processing.function_calls import OpenAISchema


class Validator(OpenAISchema):
    """Model used by LLM-powered validators."""

    is_valid: bool = Field(
        default=True,
        description="Whether the attribute is valid based on the requirements",
    )
    reason: Optional[str] = Field(
        default=None,
        description="The error message if the attribute is not valid, otherwise None",
    )
    fixed_value: Optional[str] = Field(
        default=None,
        description="If the attribute is not valid, suggest a new value for the attribute",
    )


__all__ = ["Validator"]

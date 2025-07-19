"""Validators that extend OpenAISchema for structured outputs."""

from typing import Optional

from pydantic import Field

from .function_calls import OpenAISchema


class Validator(OpenAISchema):
    """
    Validate if an attribute is correct and if not,
    return a new value with an error message
    """

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

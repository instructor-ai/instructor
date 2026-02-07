"""LLM-based validators for Pydantic field validation.

This module provides validators that use LLMs to validate field values,
with security measures to prevent prompt injection attacks.

Security Note:
    User input is wrapped in XML-style delimiters to prevent prompt injection.
    See: https://github.com/instructor-ai/instructor/issues/2056
"""

from typing import Callable
from collections.abc import Awaitable

from openai import OpenAI

from ..processing.validators import Validator
from ..core.client import Instructor, AsyncInstructor


def _format_validation_prompt(value: str, statement: str, escape: bool = True) -> str:
    """Format validation prompt with optional input escaping for security.

    When escape=True, wraps user value in XML tags to prevent prompt injection.
    This is the recommended default for security.

    Args:
        value: The user-provided value to validate
        statement: The validation rules to check against
        escape: Whether to use XML escaping (default: True)

    Returns:
        Formatted prompt string for the LLM
    """
    if escape:
        # Use XML-style delimiters to clearly separate user input from instructions
        # This prevents prompt injection by making user content visually distinct
        return (
            "Validate if the following value meets the specified rules.\n\n"
            f"<user_value>\n{value}\n</user_value>\n\n"
            f"Rules to check: {statement}\n\n"
            "Respond with is_valid=true ONLY if the content inside <user_value> tags "
            "satisfies all the rules. Otherwise respond is_valid=false with a reason."
        )
    # Legacy behavior for backward compatibility
    return f"Does `{value}` follow the rules: {statement}"


def llm_validator(
    statement: str,
    client: Instructor,
    allow_override: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    escape_user_input: bool = True,
) -> Callable[[str], str]:
    """Create a validator that uses the LLM to validate an attribute.

    This validator uses XML-style escaping by default to prevent prompt
    injection attacks. See issue #2056 for security details.

    Usage:
        ```python
        from instructor import llm_validator
        from pydantic import BaseModel, field_validator
        from typing import Annotated

        client = instructor.from_provider("openai/gpt-4o-mini")

        class User(BaseModel):
            name: Annotated[str, llm_validator(
                "The name must be a full name all lowercase",
                client=client
            )]
            age: int

        try:
            user = User(name="Jason Liu", age=20)
        except ValidationError as e:
            print(e)
        ```

    Args:
        statement: The validation rules to check the value against
        client: The Instructor client to use for validation
        allow_override: If True, return the LLM's fixed_value when validation
            fails instead of raising ValueError (default: False)
        model: The LLM model to use for validation (default: "gpt-4o-mini")
        temperature: The temperature for LLM generation (default: 0)
        escape_user_input: If True, wrap user input in XML tags to prevent
            prompt injection attacks (default: True, recommended)

    Returns:
        A callable validator function for use with Pydantic

    Raises:
        ValueError: When validation fails and allow_override is False
    """

    def llm(v: str) -> str:
        resp = client.chat.completions.create(
            response_model=Validator,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a validation model. Determine if the provided value "
                        "is valid according to the given rules. If invalid, explain why "
                        "and suggest a corrected value if possible."
                    ),
                },
                {
                    "role": "user",
                    "content": _format_validation_prompt(v, statement, escape_user_input),
                },
            ],
            model=model,
            temperature=temperature,
        )

        # Handle validation result
        if not resp.is_valid:
            # Check if we should return the fixed value instead of failing
            if allow_override and resp.fixed_value is not None:
                return resp.fixed_value
            # Raise a proper ValueError with the LLM's explanation
            raise ValueError(resp.reason)

        return v

    return llm


def async_llm_validator(
    statement: str,
    client: AsyncInstructor,
    allow_override: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    escape_user_input: bool = True,
) -> Callable[[str], Awaitable[str]]:
    """Async version of llm_validator for async validation pipelines.

    This validator uses XML-style escaping by default to prevent prompt
    injection attacks. See issue #2056 for security details.

    Usage:
        ```python
        from instructor import async_llm_validator
        from pydantic import BaseModel
        from typing import Annotated

        async_client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

        # Use with async Pydantic validators or manual async validation
        validator = async_llm_validator(
            "The name must be a full name all lowercase",
            client=async_client
        )
        validated_name = await validator("jason liu")
        ```

    Args:
        statement: The validation rules to check the value against
        client: The AsyncInstructor client to use for validation
        allow_override: If True, return the LLM's fixed_value when validation
            fails instead of raising ValueError (default: False)
        model: The LLM model to use for validation (default: "gpt-4o-mini")
        temperature: The temperature for LLM generation (default: 0)
        escape_user_input: If True, wrap user input in XML tags to prevent
            prompt injection attacks (default: True, recommended)

    Returns:
        An async callable validator function

    Raises:
        ValueError: When validation fails and allow_override is False
    """

    async def llm(v: str) -> str:
        resp = await client.chat.completions.create(
            response_model=Validator,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a validation model. Determine if the provided value "
                        "is valid according to the given rules. If invalid, explain why "
                        "and suggest a corrected value if possible."
                    ),
                },
                {
                    "role": "user",
                    "content": _format_validation_prompt(v, statement, escape_user_input),
                },
            ],
            model=model,
            temperature=temperature,
        )

        # Handle validation result
        if not resp.is_valid:
            if allow_override and resp.fixed_value is not None:
                return resp.fixed_value
            raise ValueError(resp.reason)

        return v

    return llm


def openai_moderation(client: OpenAI) -> Callable[[str], str]:
    """Validate a message using OpenAI's moderation model.

    Should only be used for monitoring inputs and outputs of OpenAI APIs.
    Other use cases are disallowed as per:
    https://platform.openai.com/docs/guides/moderation/overview

    Usage:
        ```python
        from instructor import openai_moderation
        from pydantic import BaseModel
        from typing import Annotated
        from pydantic.functional_validators import AfterValidator

        class Response(BaseModel):
            message: Annotated[str, AfterValidator(openai_moderation(client))]

        Response(message="I hate you")  # Raises ValidationError
        ```

    Args:
        client: The OpenAI client to use (must be sync)

    Returns:
        A callable validator function for use with Pydantic

    Raises:
        ValueError: When content is flagged by moderation
    """

    def validate_message_with_openai_mod(v: str) -> str:
        response = client.moderations.create(input=v)
        out = response.results[0]
        cats = out.categories.model_dump()
        if out.flagged:
            raise ValueError(
                f"`{v}` was flagged for {', '.join(cat for cat in cats if cats[cat])}"
            )

        return v

    return validate_message_with_openai_mod

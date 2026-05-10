"""LLM-backed validation helpers owned by the v2 runtime."""

from typing import Callable

from openai import OpenAI

from instructor.v2.core.client import Instructor
from instructor.v2.core.validators import Validator


def llm_validator(
    statement: str,
    client: Instructor,
    allow_override: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
) -> Callable[[str], str]:
    """Create a validator that uses an LLM to validate an attribute."""

    def llm(v: str) -> str:
        resp = client.chat.completions.create(
            response_model=Validator,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class validation model. Capable to determine if the following value is valid for the statement, if it is not, explain why and suggest a new value.",
                },
                {
                    "role": "user",
                    "content": f"Does `{v}` follow the rules: {statement}",
                },
            ],
            model=model,
            temperature=temperature,
        )

        if not resp.is_valid:
            if allow_override and resp.fixed_value is not None:
                return resp.fixed_value
            assert resp.is_valid, resp.reason

        return v

    return llm


def openai_moderation(client: OpenAI) -> Callable[[str], str]:
    """Create a validator backed by the OpenAI moderation endpoint."""

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

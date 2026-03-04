"""
Regression test for issue #1764: OpenRouter self-correction retries were not
sending the validation error back to the model correctly.

When Mode.OPENROUTER_STRUCTURED_OUTPUTS is used and a ValidationError occurs,
the reask message must instruct the model to "Correct your JSON ONLY RESPONSE"
(reask_md_json), NOT "Recall the function correctly" (reask_default), because
OpenRouter structured outputs use a JSON schema response format, not tool calls.
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from instructor.mode import Mode
from instructor.processing.response import handle_reask_kwargs
from instructor.providers.openai.utils import reask_md_json, reask_default


class StrictUser(BaseModel):
    name: str
    age: int

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("age must be positive")
        return v


def _make_mock_response(content: str) -> Mock:
    """Build a minimal mock ChatCompletion response."""
    msg = Mock()
    msg.content = content
    msg.tool_calls = None
    msg.function_call = None
    msg.role = "assistant"

    choice = Mock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = Mock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def test_openrouter_structured_outputs_uses_md_json_reask():
    """
    OPENROUTER_STRUCTURED_OUTPUTS must use reask_md_json, not reask_default.

    reask_md_json produces: "Correct your JSON ONLY RESPONSE, based on the following errors: ..."
    reask_default produces:  "Recall the function correctly, fix the errors, exceptions found ..."

    The former is correct for a JSON-schema response_format; the latter is
    meant for function/tool-calling modes and confuses models on OpenRouter.
    """
    response = _make_mock_response('{"name": "Alice", "age": -1}')
    exception = ValueError("age must be positive")

    kwargs = {
        "messages": [{"role": "user", "content": "Extract user details."}],
        "model": "openai/gpt-4o",
    }

    result = handle_reask_kwargs(
        kwargs=kwargs,
        mode=Mode.OPENROUTER_STRUCTURED_OUTPUTS,
        response=response,
        exception=exception,
        failed_attempts=None,
    )

    # There must be at least one new message appended
    assert len(result["messages"]) > 1

    last_msg = result["messages"][-1]
    assert last_msg["role"] == "user"
    content = last_msg["content"]

    # Must use the JSON-mode instruction, not the function-calling instruction
    assert "JSON" in content, (
        f"Expected JSON-specific reask message, got: {content!r}"
    )
    assert "Recall the function" not in content, (
        f"Got function-calling reask message instead of JSON reask: {content!r}"
    )


def test_openrouter_reask_includes_exception_text():
    """The reask message must include the exception detail so the model knows what to fix."""
    response = _make_mock_response('{"name": "Bob"}')
    exception = ValueError("age field is required")

    kwargs = {
        "messages": [{"role": "user", "content": "Extract user details."}],
        "model": "openai/gpt-4o",
    }

    result = handle_reask_kwargs(
        kwargs=kwargs,
        mode=Mode.OPENROUTER_STRUCTURED_OUTPUTS,
        response=response,
        exception=exception,
        failed_attempts=None,
    )

    last_msg = result["messages"][-1]
    assert "age field is required" in last_msg["content"]

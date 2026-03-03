"""
Test that llm_validator uses XML-delimited prompts and allow_override works.

Tests for the fix to issue #2056 (Finding 2: LLM Validator Injection).
"""

import pytest
from unittest.mock import Mock

from instructor.validation.llm_validators import llm_validator
from instructor.processing.validators import Validator


def _make_mock_client(
    is_valid: bool, reason: str | None = None, fixed_value: str | None = None
) -> Mock:
    """Create a mock Instructor client that returns a Validator response."""
    mock_client = Mock()
    mock_resp = Validator(is_valid=is_valid, reason=reason, fixed_value=fixed_value)
    mock_client.chat.completions.create = Mock(return_value=mock_resp)
    return mock_client


def test_llm_validator_uses_xml_delimiters():
    """Verify the prompt sent to the LLM uses XML delimiters around value and rules."""
    mock_client = _make_mock_client(is_valid=True)

    validator_fn = llm_validator(
        statement="must be lowercase",
        client=mock_client,
        model="gpt-4o-mini",
    )

    result = validator_fn("Hello World")

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args

    # Extract messages from call args
    messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", None))
    user_message = next(m for m in messages if m["role"] == "user")
    content = user_message["content"]

    # Verify XML structure
    assert "<value>" in content
    assert "</value>" in content
    assert "<rules>" in content
    assert "</rules>" in content
    assert "Hello World" in content
    assert "must be lowercase" in content

    # Verify value is within <value> tags
    value_start = content.index("<value>")
    value_end = content.index("</value>")
    assert "Hello World" in content[value_start:value_end]

    # Verify rules are within <rules> tags
    rules_start = content.index("<rules>")
    rules_end = content.index("</rules>")
    assert "must be lowercase" in content[rules_start:rules_end]

    assert result == "Hello World"


def test_llm_validator_prompt_does_not_use_raw_fstring():
    """Verify the old f-string pattern with backticks is no longer used."""
    mock_client = _make_mock_client(is_valid=True)

    validator_fn = llm_validator(
        statement="must be lowercase",
        client=mock_client,
        model="gpt-4o-mini",
    )

    validator_fn("test_value")

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", None))
    user_message = next(m for m in messages if m["role"] == "user")
    content = user_message["content"]

    # The old pattern was: f"Does `{v}` follow the rules: {statement}"
    assert "Does `" not in content
    assert "` follow the rules:" not in content


def test_llm_validator_allow_override_returns_fixed_value():
    """When allow_override=True and is_valid=False, return fixed_value."""
    mock_client = _make_mock_client(
        is_valid=False,
        reason="Not lowercase",
        fixed_value="hello world",
    )

    validator_fn = llm_validator(
        statement="must be lowercase",
        client=mock_client,
        allow_override=True,
        model="gpt-4o-mini",
    )

    result = validator_fn("Hello World")
    assert result == "hello world"


def test_llm_validator_no_override_raises_on_invalid():
    """When allow_override=False and is_valid=False, raise AssertionError."""
    mock_client = _make_mock_client(
        is_valid=False,
        reason="Not lowercase",
        fixed_value="hello world",
    )

    validator_fn = llm_validator(
        statement="must be lowercase",
        client=mock_client,
        allow_override=False,
        model="gpt-4o-mini",
    )

    with pytest.raises(AssertionError, match="Not lowercase"):
        validator_fn("Hello World")


def test_llm_validator_valid_response_returns_original():
    """When is_valid=True, return the original value unchanged."""
    mock_client = _make_mock_client(is_valid=True)

    validator_fn = llm_validator(
        statement="must be lowercase",
        client=mock_client,
        model="gpt-4o-mini",
    )

    result = validator_fn("hello world")
    assert result == "hello world"

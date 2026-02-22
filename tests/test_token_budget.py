"""
Tests for token budget enforcement in retry mechanism.

This addresses the retry amplification security issue (#2056) where retries
can cause unbounded token growth as error context is appended to messages.
"""

import pytest
from unittest.mock import Mock

from pydantic import BaseModel

import instructor
from instructor.core.exceptions import TokenBudgetExceeded
from instructor.core.retry import get_total_tokens
from instructor.mode import Mode
from openai.types.completion_usage import CompletionUsage


class User(BaseModel):
    name: str
    age: int


def _make_mock_response(content: str, usage_tokens: int):
    """Create a mock OpenAI response with real CompletionUsage for isinstance checks."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"

    mock_response.usage = CompletionUsage(
        completion_tokens=usage_tokens // 2,
        prompt_tokens=usage_tokens - usage_tokens // 2,
        total_tokens=usage_tokens,
    )

    return mock_response


def test_get_total_tokens_openai():
    """Test token extraction from OpenAI usage object."""
    usage = Mock()
    usage.total_tokens = 500
    assert get_total_tokens(usage) == 500


def test_get_total_tokens_anthropic():
    """Test token extraction from Anthropic usage object."""
    usage = Mock(spec=[])  # No attributes by default
    usage.input_tokens = 300
    usage.output_tokens = 200
    # Anthropic doesn't have total_tokens
    assert get_total_tokens(usage) == 500


def test_get_total_tokens_zero():
    """Test token extraction when no usage data is available."""
    usage = Mock(spec=[])  # No attributes
    assert get_total_tokens(usage) == 0


def test_token_budget_stops_retries():
    """Test that token_budget stops retries when cumulative tokens exceed the budget."""
    # Each response uses 3000 tokens - budget of 5000 should stop after 2 attempts
    mock_response = _make_mock_response('{"name": "John"}', usage_tokens=3000)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=10,
            token_budget=5000,
        )

    exc = exc_info.value
    assert exc.total_tokens > 5000
    assert exc.token_budget == 5000
    assert exc.n_attempts <= 10  # Should have stopped before max_retries


def test_token_budget_not_triggered_when_within_budget():
    """Test that token_budget does not interfere when usage is within budget."""
    mock_response = _make_mock_response('{"name": "John", "age": 30}', usage_tokens=100)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    # Should succeed without raising TokenBudgetExceeded
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=User,
        messages=[{"role": "user", "content": "test"}],
        max_retries=3,
        token_budget=50000,
    )

    assert result.name == "John"
    assert result.age == 30


def test_token_budget_none_disables_check():
    """Test that omitting token_budget does not change existing behavior."""
    # Invalid JSON should still exhaust retries normally
    mock_response = _make_mock_response("invalid json {", usage_tokens=1000)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    from instructor.core.exceptions import InstructorRetryException

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
            # No token_budget - should exhaust retries normally
        )

    assert exc_info.value.n_attempts == 2


def test_token_budget_exceeded_exception_attributes():
    """Test that TokenBudgetExceeded has the expected attributes."""
    exc = TokenBudgetExceeded(
        total_tokens=6000,
        token_budget=5000,
        last_completion=None,
        n_attempts=3,
        total_usage=None,
    )

    assert exc.total_tokens == 6000
    assert exc.token_budget == 5000
    assert exc.n_attempts == 3
    assert "6000" in str(exc)
    assert "5000" in str(exc)


def test_token_budget_exceeded_is_instructor_error():
    """Test that TokenBudgetExceeded inherits from InstructorError."""
    from instructor.core.exceptions import InstructorError

    assert issubclass(TokenBudgetExceeded, InstructorError)

    with pytest.raises(InstructorError):
        raise TokenBudgetExceeded(
            total_tokens=100,
            token_budget=50,
            n_attempts=1,
        )

"""
Test that token_budget parameter stops retries when token usage exceeds budget.

Tests for the fix to issue #2056 (Finding 1: Retry Amplification).
"""

import pytest
from unittest.mock import Mock
from pydantic import BaseModel

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor.mode import Mode
from instructor.utils import get_total_tokens
from openai.types.completion_usage import CompletionUsage
from typing import cast


class User(BaseModel):
    name: str
    age: int


def _make_mock_response(content: str, total_tokens: int) -> Mock:
    """Create a mock response with usage data that will fail validation."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = CompletionUsage(
        completion_tokens=total_tokens // 2,
        prompt_tokens=total_tokens - total_tokens // 2,
        total_tokens=total_tokens,
    )
    return mock_response


def test_token_budget_stops_retries():
    """Token budget should stop retries when cumulative usage exceeds budget."""
    # Each call uses 500 tokens; budget is 800
    # After attempt 1: 500 tokens used (under budget, retry)
    # Before attempt 3: 1000 tokens used (over 800 budget, stop)
    mock_response = _make_mock_response('{"name": "John"}', total_tokens=500)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=5,
            token_budget=800,
        )

    exception = cast(InstructorRetryException, exc_info.value)
    # Should stop before attempt 3 (after 2 calls = 1000 tokens > 800)
    assert exception.n_attempts == 2
    assert exception.total_usage is not None


def test_token_budget_none_allows_all_retries():
    """When token_budget is None (default), all retries should proceed."""
    mock_response = _make_mock_response('{"name": "John"}', total_tokens=500)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=3,
        )

    exception = cast(InstructorRetryException, exc_info.value)
    assert exception.n_attempts == 3


def test_token_budget_high_allows_all_retries():
    """A very high token budget should not stop any retries."""
    mock_response = _make_mock_response('{"name": "John"}', total_tokens=100)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=3,
            token_budget=999999,
        )

    exception = cast(InstructorRetryException, exc_info.value)
    assert exception.n_attempts == 3


def test_token_budget_exception_includes_usage():
    """Token budget exception should include total_usage data."""
    mock_response = _make_mock_response('{"name": "John"}', total_tokens=500)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=10,
            token_budget=800,
        )

    exception = cast(InstructorRetryException, exc_info.value)
    assert exception.total_usage is not None
    assert get_total_tokens(exception.total_usage) > 800


def test_get_total_tokens_openai():
    """get_total_tokens should extract total_tokens from OpenAI usage."""
    usage = CompletionUsage(
        completion_tokens=100,
        prompt_tokens=200,
        total_tokens=300,
    )
    assert get_total_tokens(usage) == 300


def test_get_total_tokens_none():
    """get_total_tokens should return 0 for None."""
    assert get_total_tokens(None) == 0


def test_get_total_tokens_unknown():
    """get_total_tokens should return 0 for unrecognized objects."""
    assert get_total_tokens("not a usage object") == 0


def test_get_total_tokens_anthropic():
    """get_total_tokens should sum input_tokens + output_tokens for Anthropic usage."""
    try:
        from anthropic.types import Usage as AnthropicUsage
    except ImportError:
        pytest.skip("anthropic not installed")

    usage = AnthropicUsage(
        input_tokens=150,
        output_tokens=250,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    assert get_total_tokens(usage) == 400


def test_token_budget_zero_allows_one_attempt():
    """token_budget=0 should allow the first attempt but stop before the second."""
    mock_response = _make_mock_response('{"name": "John"}', total_tokens=100)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=5,
            token_budget=0,
        )

    exception = cast(InstructorRetryException, exc_info.value)
    # First attempt runs (100 tokens), budget check triggers before attempt 2
    assert exception.n_attempts == 1

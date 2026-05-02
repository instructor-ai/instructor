"""
Test that token_budget stops retries when cumulative token usage exceeds the budget.

This tests the mitigation for retry amplification (issue #2056).
"""

from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, field_validator

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor.mode import Mode


class StrictAge(BaseModel):
    name: str
    age: int

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("age must be positive")
        return v


def _make_completion(content: str, usage_tokens: int) -> ChatCompletion:
    return ChatCompletion(
        id="test",
        model="gpt-4",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=usage_tokens // 2,
            completion_tokens=usage_tokens // 2,
            total_tokens=usage_tokens,
        ),
    )


def test_token_budget_stops_retries():
    """Retries stop when cumulative token usage exceeds token_budget."""
    # Each call uses 500 tokens, budget is 800 -- should stop after 2 attempts
    bad_response = _make_completion('{"name": "Alice", "age": -1}', usage_tokens=500)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=bad_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4",
            response_model=StrictAge,
            messages=[{"role": "user", "content": "give me a user"}],
            max_retries=10,
            token_budget=800,
        )

    # Should have stopped well before 10 retries due to budget
    assert exc_info.value.n_attempts <= 3


def test_token_budget_not_set_retries_normally():
    """Without token_budget, all retries are exhausted."""
    bad_response = _make_completion('{"name": "Alice", "age": -1}', usage_tokens=500)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=bad_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(
            model="gpt-4",
            response_model=StrictAge,
            messages=[{"role": "user", "content": "give me a user"}],
            max_retries=5,
        )

    assert exc_info.value.n_attempts == 5


def test_token_budget_success_before_limit():
    """If validation succeeds before budget is hit, result is returned normally."""
    good_response = _make_completion('{"name": "Alice", "age": 30}', usage_tokens=200)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=good_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    result = client.chat.completions.create(
        model="gpt-4",
        response_model=StrictAge,
        messages=[{"role": "user", "content": "give me a user"}],
        max_retries=5,
        token_budget=1000,
    )

    assert result.name == "Alice"
    assert result.age == 30

"""Tests for token usage tracking and budget enforcement.

Verifies that:
1. Total token usage is attached to successful responses (_total_usage)
2. The completion:usage hook fires after each attempt with cumulative usage
3. token_budget parameter raises TokenBudgetExceeded when exceeded
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import TokenBudgetExceeded
from instructor.v2.core.hooks import HookName, Hooks
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import retry_sync_v2


class User(BaseModel):
    name: str
    age: int


def _make_openai_response(content: str, usage_tokens: int = 100):
    """Create a mock OpenAI-like response with usage data."""
    from openai.types.completion_usage import (
        CompletionTokensDetails,
        CompletionUsage,
        PromptTokensDetails,
    )

    response = MagicMock()
    response.usage = CompletionUsage(
        completion_tokens=usage_tokens // 2,
        prompt_tokens=usage_tokens // 2,
        total_tokens=usage_tokens,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    )
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.tool_calls = [MagicMock()]
    response.choices[0].message.tool_calls[0].function = MagicMock()
    response.choices[0].message.tool_calls[0].function.arguments = (
        '{"name": "Alice", "age": 30}'
    )
    return response


@patch("instructor.v2.core.retry.mode_registry")
@patch("instructor.v2.core.retry.RegistryValidationMixin")
def test_total_usage_attached_on_success(mock_validation, mock_registry):
    """Successful extraction should have _total_usage attached to the result."""
    response = _make_openai_response('{"name": "Alice", "age": 30}', usage_tokens=150)
    func = MagicMock(return_value=response)

    handlers = MagicMock()
    handlers.response_parser.return_value = User(name="Alice", age=30)
    mock_registry.get_handlers.return_value = handlers

    result = retry_sync_v2(
        func=func,
        response_model=User,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=3,
        args=(),
        kwargs={},
        strict=True,
    )

    assert result.name == "Alice"
    assert result.age == 30
    assert hasattr(result, "_total_usage")
    assert result._total_usage.total_tokens == 150


@patch("instructor.v2.core.retry.mode_registry")
@patch("instructor.v2.core.retry.RegistryValidationMixin")
def test_usage_hook_fires_after_each_attempt(mock_validation, mock_registry):
    """completion:usage hook should fire with cumulative usage after each attempt."""
    response = _make_openai_response('{"name": "Bob", "age": 25}', usage_tokens=200)
    func = MagicMock(return_value=response)

    handlers = MagicMock()
    handlers.response_parser.return_value = User(name="Bob", age=25)
    mock_registry.get_handlers.return_value = handlers

    hooks = Hooks()
    usage_events: list = []

    def on_usage(usage, *, attempt_number=0):
        usage_events.append({"tokens": usage.total_tokens, "attempt": attempt_number})

    hooks.on(HookName.COMPLETION_USAGE, on_usage)

    retry_sync_v2(
        func=func,
        response_model=User,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=3,
        args=(),
        kwargs={},
        strict=True,
        hooks=hooks,
    )

    assert len(usage_events) == 1
    assert usage_events[0]["tokens"] == 200
    assert usage_events[0]["attempt"] == 1


@patch("instructor.v2.core.retry.mode_registry")
@patch("instructor.v2.core.retry.RegistryValidationMixin")
def test_token_budget_raises_when_exceeded(mock_validation, mock_registry):
    """token_budget should raise TokenBudgetExceeded when cumulative tokens exceed it."""
    response = _make_openai_response('{"name": "X", "age": 1}', usage_tokens=500)
    func = MagicMock(return_value=response)

    handlers = MagicMock()
    handlers.response_parser.return_value = User(name="X", age=1)
    mock_registry.get_handlers.return_value = handlers

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        retry_sync_v2(
            func=func,
            response_model=User,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=3,
            args=(),
            kwargs={},
            strict=True,
            token_budget=100,
        )

    assert exc_info.value.budget == 100
    assert exc_info.value.n_attempts == 1
    assert exc_info.value.total_usage.total_tokens == 500


@patch("instructor.v2.core.retry.mode_registry")
@patch("instructor.v2.core.retry.RegistryValidationMixin")
def test_token_budget_none_does_not_limit(mock_validation, mock_registry):
    """When token_budget is None, no budget enforcement should occur."""
    response = _make_openai_response('{"name": "Y", "age": 99}', usage_tokens=9999)
    func = MagicMock(return_value=response)

    handlers = MagicMock()
    handlers.response_parser.return_value = User(name="Y", age=99)
    mock_registry.get_handlers.return_value = handlers

    result = retry_sync_v2(
        func=func,
        response_model=User,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=3,
        args=(),
        kwargs={},
        strict=True,
        token_budget=None,
    )

    assert result.name == "Y"
    assert result._total_usage.total_tokens == 9999


@patch("instructor.v2.core.retry.mode_registry")
@patch("instructor.v2.core.retry.RegistryValidationMixin")
def test_token_budget_allows_under_budget(mock_validation, mock_registry):
    """Requests under the token budget should succeed normally."""
    response = _make_openai_response('{"name": "Z", "age": 5}', usage_tokens=50)
    func = MagicMock(return_value=response)

    handlers = MagicMock()
    handlers.response_parser.return_value = User(name="Z", age=5)
    mock_registry.get_handlers.return_value = handlers

    result = retry_sync_v2(
        func=func,
        response_model=User,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=3,
        args=(),
        kwargs={},
        strict=True,
        token_budget=1000,
    )

    assert result.name == "Z"
    assert result._total_usage.total_tokens == 50


@patch("instructor.v2.core.retry.mode_registry")
@patch("instructor.v2.core.retry.RegistryValidationMixin")
def test_usage_hook_string_registration(mock_validation, mock_registry):
    """completion:usage hook should work with string registration."""
    response = _make_openai_response('{"name": "C", "age": 10}', usage_tokens=77)
    func = MagicMock(return_value=response)

    handlers = MagicMock()
    handlers.response_parser.return_value = User(name="C", age=10)
    mock_registry.get_handlers.return_value = handlers

    hooks = Hooks()
    called = []

    def handler(usage, **kwargs):
        called.append(usage.total_tokens)

    hooks.on("completion:usage", handler)

    retry_sync_v2(
        func=func,
        response_model=User,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=1,
        args=(),
        kwargs={},
        strict=True,
        hooks=hooks,
    )

    assert called == [77]

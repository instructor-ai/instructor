"""Regression tests for issue #2417.

``client.create()`` (and the underlying v2 handlers) must treat the caller's
``messages`` list as a read-only input. Before the fix, the shared OpenAI
handlers only performed a shallow ``kwargs.copy()`` in ``prepare_request``, so
the ``messages`` value was still the caller's own list. When a validation
failure triggered the reask/retry path, synthetic messages were appended to
that list in place -- corrupting the caller's conversation state even on a
successful call.

See https://github.com/567-labs/instructor/issues/2417 (Category 1).
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from instructor import Mode, Provider
from instructor.v2.core.retry import retry_sync_v2
from instructor.v2.core.registry import mode_registry

OPENAI_COMPAT_PROVIDERS = (
    Provider.ANYSCALE,
    Provider.TOGETHER,
    Provider.DATABRICKS,
    Provider.DEEPSEEK,
    Provider.OPENROUTER,
    Provider.GROQ,
    Provider.FIREWORKS,
    Provider.CEREBRAS,
)


class Answer(BaseModel):
    answer: float


def _tool_call_response(arguments: str) -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_0",
                type="function",
                function=Function(name="Answer", arguments=arguments),
            )
        ],
    )
    choice = {"index": 0, "message": message, "finish_reason": "stop"}
    return ChatCompletion(
        id="cmpl-issue-2417",
        choices=[choice],  # type: ignore[arg-type]
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
    )


@pytest.mark.parametrize(
    "mode",
    [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
)
def test_openai_prepare_request_does_not_alias_caller_messages(mode: Mode) -> None:
    caller_messages = [{"role": "user", "content": "What is 2+2?"}]
    original = [dict(m) for m in caller_messages]

    # PARALLEL_TOOLS requires an Iterable-wrapped response model.
    response_model: Any = Iterable[Answer] if mode is Mode.PARALLEL_TOOLS else Answer

    _, out = mode_registry.get_handlers(Provider.OPENAI, mode).request_handler(
        response_model, {"messages": caller_messages}
    )

    # A new list object must be used for the outgoing request.
    assert out["messages"] is not caller_messages
    # The caller's list (and its contents) must be completely untouched.
    assert caller_messages == original


@pytest.mark.parametrize("provider", OPENAI_COMPAT_PROVIDERS)
def test_compat_providers_reuse_fixed_openai_tools_handler(provider: Provider) -> None:
    # The TOOLS handler is a single shared class registered for every OpenAI
    # compatible provider, so the Category 1 fix covers them all at once.
    caller_messages = [{"role": "user", "content": "What is 2+2?"}]
    original = [dict(m) for m in caller_messages]

    _, out = mode_registry.get_handlers(provider, Mode.TOOLS).request_handler(
        Answer, {"messages": caller_messages}
    )

    assert out["messages"] is not caller_messages
    assert caller_messages == original


def test_reask_does_not_mutate_caller_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end reproduction of issue #2417 through ``retry_sync_v2``.

    A validation failure on the first attempt forces a reask; the synthetic
    messages added by the reask path must land on a *copy* of the caller's
    list, leaving the original untouched.
    """
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        lambda _provider: SimpleNamespace(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        ),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        lambda _response, total_usage: total_usage,
    )

    caller_messages = [{"role": "user", "content": "What is 2+2?"}]
    attempts: list[int] = []

    def fake_func(*_args: Any, **_kwargs: Any) -> ChatCompletion:
        attempts.append(1)
        # First attempt: invalid tool-call payload -> validation error -> reask.
        # Second attempt: valid payload -> parsed successfully.
        if len(attempts) == 1:
            return _tool_call_response('{"answer": "not-a-number"}')
        return _tool_call_response('{"answer": 4}')

    # Run prepare_request exactly as the patched client does, so the caller's
    # messages list is the one we want to prove stays untouched.
    handlers = mode_registry.get_handlers(Provider.OPENAI, Mode.TOOLS)
    _, prepared = handlers.request_handler(Answer, {"messages": caller_messages})

    result = retry_sync_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=Retrying(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        ),
        args=(),
        kwargs=prepared,
        strict=True,
        hooks=None,
    )

    assert isinstance(result, Answer)
    assert result.answer == 4.0
    # The caller's list must be unchanged by the reask/retry path.
    assert caller_messages == [{"role": "user", "content": "What is 2+2?"}]

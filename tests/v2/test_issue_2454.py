"""Regression test for issue #2454: `cache=` silently stops caching any
request that needed at least one retry.

When the first call to a patched create function triggers a validation
failure (and thus a reask), the successful response from the retry
attempt must be cached under the same key that an identical follow-up
call will look up. Before the fix, the reask handler's in-place
mutation of the shared `messages` list diverged the lookup key
(computed before retry) from the store key (computed after retry,
when `new_kwargs["messages"]` had been mutated to include the
validation-feedback turns). The net effect: any request that ever
needed a retry was effectively un-cached, forcing every identical
follow-up call back through the real LLM.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from instructor.cache import AutoCache
from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch_v2
from instructor.v2.core.providers import Provider


class Answer(BaseModel):
    name: str
    age: int


def _make_tool_call_response(arguments: str, call_id: str) -> ChatCompletion:
    tool_call = ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name="Answer", arguments=arguments),
    )
    message = ChatCompletionMessage(
        role="assistant", content=None, tool_calls=[tool_call]
    )
    choice = Choice(index=0, message=message, finish_reason="tool_calls", logprobs=None)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
    )


def test_cache_survives_reask_loop_sync() -> None:
    """Forces a validation failure on the first attempt, then a success
    on the second. The second identical client.create() must hit the
    cache and must NOT call the underlying LLM again.
    """
    call_count = {"n": 0}

    def fake_create(*_args: Any, **_kwargs: Any) -> ChatCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Validation fails: `age` is missing
            return _make_tool_call_response('{"name": "Ada"}', "call_1")
        # Validation succeeds
        return _make_tool_call_response('{"name": "Ada", "age": 37}', "call_2")

    create_fn = patch_v2(fake_create, provider=Provider.OPENAI, mode=Mode.TOOLS)
    cache = AutoCache(maxsize=10)

    messages = [{"role": "user", "content": "Ada is 37 years old"}]

    r1 = create_fn(
        model="gpt-4o-mini",
        messages=messages,
        response_model=Answer,
        max_retries=2,
        cache=cache,
    )
    assert isinstance(r1, Answer)
    assert r1.name == "Ada"
    assert r1.age == 37
    # First call needed 1 retry -> 2 LLM calls
    assert call_count["n"] == 2

    # Second, IDENTICAL call. Must hit the cache, must NOT call fake_create.
    r2 = create_fn(
        model="gpt-4o-mini",
        messages=messages,
        response_model=Answer,
        max_retries=2,
        cache=cache,
    )
    assert isinstance(r2, Answer)
    assert r2.name == "Ada"
    assert r2.age == 37
    # If this assertion fails, the cache key computed at store time no
    # longer matches the key computed at lookup time -- i.e. the reask
    # handler has mutated `new_kwargs["messages"]` in place between the
    # two key computations. See issue #2454.
    assert call_count["n"] == 2, (
        "Cache miss after retry: second identical call re-invoked the LLM "
        f"(call_count={call_count['n']}). The cache lookup/store keys "
        "diverged -- see issue #2454."
    )


@pytest.mark.asyncio
async def test_cache_survives_reask_loop_async() -> None:
    """Same regression as the sync test, but exercising the async wrapper
    (`_create_async_wrapper` -> `retry_async_v2`). The cache key
    divergence affects both wrappers because they share the same
    `new_kwargs` aliasing pattern.
    """
    call_count = {"n": 0}

    async def fake_create_async(*_args: Any, **_kwargs: Any) -> ChatCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_tool_call_response('{"name": "Ada"}', "call_1")
        return _make_tool_call_response('{"name": "Ada", "age": 37}', "call_2")

    create_fn = patch_v2(fake_create_async, provider=Provider.OPENAI, mode=Mode.TOOLS)
    cache = AutoCache(maxsize=10)

    messages = [{"role": "user", "content": "Ada is 37 years old"}]

    r1 = await create_fn(
        model="gpt-4o-mini",
        messages=messages,
        response_model=Answer,
        max_retries=2,
        cache=cache,
    )
    assert isinstance(r1, Answer)
    assert r1.age == 37
    assert call_count["n"] == 2

    r2 = await create_fn(
        model="gpt-4o-mini",
        messages=messages,
        response_model=Answer,
        max_retries=2,
        cache=cache,
    )
    assert isinstance(r2, Answer)
    assert r2.age == 37
    assert call_count["n"] == 2, (
        "Async cache miss after retry: second identical call re-invoked "
        f"the LLM (call_count={call_count['n']}). See issue #2454."
    )

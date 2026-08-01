import types

import instructor
from instructor.cache import AutoCache
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, field_validator  # type: ignore[import-not-found]


def test_auto_cache_prevents_duplicate_provider_calls(monkeypatch):
    _ = monkeypatch  # unused fixture for parity with other tests
    """Ensure that AutoCache prevents duplicate provider calls via patch layer."""

    class User(BaseModel):
        name: str = Field(...)

    call_counter = {"n": 0}

    # Fake provider completion function mimicking minimal OpenAI chat response
    def fake_completion(*_args, **_kwargs):  # noqa: D401, ANN001
        call_counter["n"] += 1
        content = User(name="cached").model_dump_json()
        # Return minimal ChatCompletion-like object
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage={},
        )

    # Create Instructor client using from_litellm so we go through patch stack
    cache = AutoCache(maxsize=10)
    client = instructor.from_litellm(fake_completion, mode=instructor.Mode.JSON)

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]

    # First call – provider should be invoked
    _ = client.create(messages=list(messages), response_model=User, cache=cache)
    assert call_counter["n"] == 1

    # Second call with identical inputs – should hit cache, no new provider call
    _ = client.create(messages=list(messages), response_model=User, cache=cache)
    assert call_counter["n"] == 1, "Cache miss – provider was called again"


def test_auto_cache_prevents_duplicate_calls_after_a_retry(monkeypatch):
    _ = monkeypatch
    """Regression test: a call that needed a retry must still be cacheable.

    Reask handlers append/extend the request's messages list in place. If that
    list is the same object patch.py reads again to compute the cache store key,
    the store key ends up different from the lookup key computed before the
    retry, so a later, identical call never hits the cache.
    """

    class Answer(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def must_be_positive(cls, v: int) -> int:
            if v < 0:
                raise ValueError("value must be positive")
            return v

    call_counter = {"n": 0}

    def fake_completion(*_args, **_kwargs):
        call_counter["n"] += 1
        # First call ever returns an invalid value, forcing exactly one retry.
        value = -5 if call_counter["n"] == 1 else 42
        content = Answer.model_construct(value=value).model_dump_json()
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage={},
        )

    cache = AutoCache(maxsize=10)
    client = instructor.from_litellm(fake_completion, mode=instructor.Mode.JSON)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "what is 6 times 7?"}
    ]

    result1 = client.create(
        messages=list(messages), response_model=Answer, max_retries=2, cache=cache
    )
    assert result1.value == 42
    assert call_counter["n"] == 2, "First call should need exactly one retry"

    result2 = client.create(
        messages=list(messages), response_model=Answer, max_retries=2, cache=cache
    )
    assert result2.value == 42
    assert call_counter["n"] == 2, (
        "Second, identical call should hit the cache instead of calling the provider again"
    )

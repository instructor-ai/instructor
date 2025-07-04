import types

import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


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

    messages = [{"role": "user", "content": "hello"}]

    # First call – provider should be invoked
    _ = client.create(messages=list(messages), response_model=User, cache=cache)
    assert call_counter["n"] == 1

    # Second call with identical inputs – should hit cache, no new provider call
    _ = client.create(messages=list(messages), response_model=User, cache=cache)
    assert call_counter["n"] == 1, "Cache miss – provider was called again"

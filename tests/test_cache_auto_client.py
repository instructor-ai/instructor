import sys, types
import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


def test_from_provider_cache_passthrough(monkeypatch):
    """Ensure cache provided to from_provider is used for create calls."""

    # ------------------------------------------------------------------
    # Build dummy openai module with minimal Chat API
    # ------------------------------------------------------------------
    dummy_openai = types.ModuleType("openai")

    class _DummyChatCompletions:
        @staticmethod
        def create(*args, **kwargs):  # noqa: D401, ANN001
            # Count how many times the actual provider is invoked
            counter["n"] += 1
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message={"content": User(name="hi").model_dump_json()})
                ],
                usage={},
            )

    class _DummyChat:
        completions = _DummyChatCompletions

    class _DummyOpenAI:
        def __init__(self, *a, **k):
            self.chat = _DummyChat()

    dummy_openai.OpenAI = _DummyOpenAI  # type: ignore[attr-defined]
    dummy_openai.AsyncOpenAI = _DummyOpenAI  # type: ignore[attr-defined]
    sys.modules["openai"] = dummy_openai

    # ------------------------------------------------------------------
    # Prepare cache and client
    # ------------------------------------------------------------------
    cache = AutoCache(maxsize=10)

    class User(BaseModel):
        name: str = Field(...)

    global counter
    counter = {"n": 0}

    client = instructor.from_provider("openai/gpt-3.5-turbo", cache=cache)

    messages = [{"role": "user", "content": "hello"}]

    # First call uses provider (counter == 1)
    _ = client.create(messages=messages, response_model=User)
    assert counter["n"] == 1

    # Second identical call should hit cache (counter unchanged)
    _ = client.create(messages=messages, response_model=User)
    assert counter["n"] == 1
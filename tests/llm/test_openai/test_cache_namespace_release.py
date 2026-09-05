from typing import Any

import openai
import pytest
from pydantic import BaseModel

import instructor
from instructor.cache import AutoCache


class Answer(BaseModel):
    value: int


class CountingCache(AutoCache):
    def __init__(self) -> None:
        super().__init__()
        self.misses = 0

    def get(self, key: str) -> Any:
        result = super().get(key)
        if result is None:
            self.misses += 1
        return result


def request(cache: CountingCache) -> dict[str, Any]:
    return {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Return the integer 42."}],
        "response_model": Answer,
        "cache": cache,
        "temperature": 0,
    }


def test_live_sync_cache_namespaces() -> None:
    cache = CountingCache()
    with openai.OpenAI() as sdk_a, openai.OpenAI() as sdk_b:
        a, b = instructor.from_openai(sdk_a), instructor.from_openai(sdk_b)
        kwargs = request(cache)
        assert a.create(**kwargs).value == 42
        assert a.create(**kwargs).value == 42
        assert cache.misses == 1
        assert b.create(**kwargs).value == 42
        assert cache.misses == 2
        assert a.create(**kwargs, cache_namespace="shared-release-test").value == 42
        assert b.create(**kwargs, cache_namespace="shared-release-test").value == 42
        assert cache.misses == 3


@pytest.mark.asyncio
async def test_live_async_cache_namespaces() -> None:
    cache = CountingCache()
    async with openai.AsyncOpenAI() as sdk_a, openai.AsyncOpenAI() as sdk_b:
        a, b = instructor.from_openai(sdk_a), instructor.from_openai(sdk_b)
        kwargs = request(cache)
        assert (await a.create(**kwargs)).value == 42
        assert (await a.create(**kwargs)).value == 42
        assert cache.misses == 1
        assert (await b.create(**kwargs)).value == 42
        assert cache.misses == 2
        assert (
            await a.create(**kwargs, cache_namespace="shared-release-test")
        ).value == 42
        assert (
            await b.create(**kwargs, cache_namespace="shared-release-test")
        ).value == 42
        assert cache.misses == 3

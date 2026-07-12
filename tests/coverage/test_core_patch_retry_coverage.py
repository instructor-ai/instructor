from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from tenacity import AsyncRetrying, Retrying

from instructor.cache import BaseCache
from instructor.v2.core.errors import (
    IncompleteOutputException,
    InstructorRetryException,
)
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch, patch_v2
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import (
    _finalize_parsed_response,
    retry_async,
    retry_async_v2,
    retry_sync,
    retry_sync_v2,
)
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.response_list import ListResponse
from instructor.v2.dsl.simple_type import ModelAdapter


class Answer(BaseModel):
    value: int


def _completion(value: Any, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content=json.dumps({"value": value})
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


class RecordingCache(BaseCache):
    def __init__(self, fail_on_store: bool = False) -> None:
        self.values: dict[str, Any] = {}
        self.ttls: list[int | None] = []
        self.fail_on_store = fail_on_store

    def get(self, key: str) -> Any | None:
        return self.values.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.ttls.append(ttl)
        if self.fail_on_store:
            raise ModuleNotFoundError("optional cache backend is unavailable")
        self.values[key] = value


def test_patch_requires_a_target_and_supports_a_create_callable() -> None:
    with pytest.raises(ValueError, match="Either client or create must be provided"):
        patch()

    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion(8)

    patched = patch(create=create, provider=Provider.OPENAI, mode=Mode.JSON)
    result = patched(
        response_model=Answer,
        model="test-model",
        messages=[{"role": "user", "content": "give me eight"}],
    )

    assert result.value == 8
    assert calls[0]["model"] == "test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_sync_patch_ignores_a_missing_optional_cache_backend() -> None:
    calls = 0

    def create(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _completion(11)

    cache = RecordingCache(fail_on_store=True)
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")

    result = patched(
        response_model=Answer,
        messages=[{"role": "user", "content": "eleven"}],
        cache=cache,
        cache_ttl="not-an-integer",
    )

    assert result.value == 11
    assert calls == 1
    assert cache.ttls == [None]


@pytest.mark.asyncio
async def test_async_patch_stores_a_cache_miss_and_returns_a_cache_hit() -> None:
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion(13)

    cache = RecordingCache()
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON, default_model="default")
    request = {"role": "user", "content": "thirteen"}

    first = await patched(
        response_model=Answer,
        messages=[dict(request)],
        cache=cache,
        cache_ttl=45,
    )
    second = await patched(
        response_model=Answer,
        messages=[dict(request)],
        cache=cache,
        cache_ttl=45,
    )

    assert first.value == 13
    assert second.value == 13
    assert len(calls) == 1
    assert calls[0]["model"] == "default"
    assert cache.ttls == [45]
    assert len(cache.values) == 1


@pytest.mark.asyncio
async def test_async_patch_ignores_a_missing_optional_cache_backend() -> None:
    calls = 0

    async def create(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _completion(17)

    cache = RecordingCache(fail_on_store=True)
    patched = patch_v2(create, Provider.OPENAI, Mode.JSON)

    result = await patched(
        response_model=Answer,
        model="test-model",
        messages=[{"role": "user", "content": "seventeen"}],
        cache=cache,
        cache_ttl=10,
    )

    assert result.value == 17
    assert calls == 1
    assert cache.ttls == [10]


def test_finalize_converts_iterable_models_and_unwraps_simple_types() -> None:
    response = _completion(0)
    iterable_type = IterableModel(Answer)
    iterable = iterable_type(tasks=[Answer(value=1), Answer(value=2)])

    finalized = _finalize_parsed_response(iterable, response)

    assert isinstance(finalized, ListResponse)
    assert list(finalized) == [Answer(value=1), Answer(value=2)]
    assert finalized.get_raw_response() is response

    adapted = ModelAdapter[int](content=19)
    assert _finalize_parsed_response(adapted, response) == 19


def test_retry_sync_wrapper_forwards_args_timeout_stream_and_non_strict_mode() -> None:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create(*args: Any, **kwargs: Any) -> SimpleNamespace:
        seen.append((args, kwargs))
        return _completion("23")

    result = retry_sync(
        func=create,
        response_model=Answer,
        args=("request-id",),
        kwargs={
            "messages": [{"role": "user", "content": "twenty-three"}],
            "timeout": 30,
            "stream": True,
        },
        strict=False,
        mode=Mode.JSON,
        provider=Provider.OPENAI,
    )

    assert result == Answer(value=23)
    assert seen == [
        (
            ("request-id",),
            {
                "messages": [{"role": "user", "content": "twenty-three"}],
                "timeout": 30,
                "stream": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_retry_async_wrapper_reasks_emits_parse_hook_and_forwards_args() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    parse_errors: list[tuple[Exception, dict[str, Any]]] = []
    responses = iter([_completion("invalid"), _completion("29")])

    async def create(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append((args, kwargs))
        return next(responses)

    hooks = Hooks()
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )

    result = await retry_async(
        func=create,
        response_model=Answer,
        args=("request-id",),
        kwargs={
            "messages": [{"role": "user", "content": "twenty-nine"}],
            "timeout": 30.0,
            "stream": True,
        },
        max_retries=1,
        strict=False,
        mode=Mode.JSON,
        provider=Provider.OPENAI,
        hooks=hooks,
    )

    assert result == Answer(value=29)
    assert len(calls) == 2
    assert calls[0][0] == ("request-id",)
    assert calls[0][1]["timeout"] == 30.0
    assert calls[0][1]["stream"] is True
    assert calls[1][1]["messages"][-1]["role"] == "user"
    assert "Correct your JSON ONLY RESPONSE" in calls[1][1]["messages"][-1]["content"]
    assert len(parse_errors) == 1
    assert parse_errors[0][1] == {
        "attempt_number": 1,
        "max_attempts": 2,
        "is_last_attempt": False,
    }


def test_retry_sync_does_not_retry_an_incomplete_streaming_response() -> None:
    response = _completion("partial", finish_reason="length")
    calls = 0

    def create(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return response

    with pytest.raises(IncompleteOutputException) as exc_info:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [], "stream": True},
            strict=True,
        )

    assert exc_info.value.last_completion is response
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_async_does_not_retry_an_incomplete_streaming_response() -> None:
    response = _completion("partial", finish_reason="length")
    calls = 0

    async def create(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return response

    with pytest.raises(IncompleteOutputException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [], "stream": True},
            strict=True,
        )

    assert exc_info.value.last_completion is response
    assert calls == 1


class NoAttemptsRetrying(Retrying):
    def __iter__(self) -> Iterator[Any]:
        return iter(())


class NoAttemptsAsyncRetrying(AsyncRetrying):
    def __aiter__(self) -> NoAttemptsAsyncRetrying:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


def test_retry_sync_reports_a_policy_that_yields_no_attempts() -> None:
    calls = 0

    def create(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _completion(31)

    with pytest.raises(InstructorRetryException, match="Unknown error") as exc_info:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=NoAttemptsRetrying(),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "unused"}]},
            strict=True,
        )

    assert exc_info.value.n_attempts == 0
    assert exc_info.value.create_kwargs == {
        "messages": [{"role": "user", "content": "unused"}]
    }
    assert calls == 0


@pytest.mark.asyncio
async def test_retry_async_reports_a_policy_that_yields_no_attempts() -> None:
    calls = 0

    async def create(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _completion(37)

    with pytest.raises(InstructorRetryException, match="Unknown error") as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=NoAttemptsAsyncRetrying(),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "unused"}]},
            strict=True,
        )

    assert exc_info.value.n_attempts == 0
    assert exc_info.value.create_kwargs == {
        "messages": [{"role": "user", "content": "unused"}]
    }
    assert calls == 0

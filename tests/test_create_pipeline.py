from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor.cache import AutoCache, make_cache_key, store_cached_response
from instructor.core.create_pipeline import (
    CreatePipelineState,
    cache_lookup_middleware,
    cache_store_middleware,
    context_middleware,
    retry_async_middleware,
    retry_sync_middleware,
    templating_middleware,
)
from instructor.core.exceptions import ConfigurationError
from instructor.mode import Mode


class DummyModel(BaseModel):
    value: str


def make_state(**overrides):
    state = CreatePipelineState(
        func=lambda *args, **kwargs: None,
        mode=Mode.JSON,
        args=(),
        kwargs={"messages": [{"role": "user", "content": "Hello {{name}}"}], "model": "gpt"},
        response_model=DummyModel,
        validation_context=None,
        context=None,
        max_retries=1,
        strict=True,
        hooks=None,
        cache=None,
        cache_ttl=None,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_context_middleware_promotes_validation_context():
    state = make_state(validation_context={"foo": "bar"})
    context_middleware(state)
    assert state.context == {"foo": "bar"}
    assert state.validation_context is None


def test_context_middleware_conflict_raises():
    state = make_state(context={"foo": 1}, validation_context={"bar": 2})
    with pytest.raises(ConfigurationError):
        context_middleware(state)


def test_templating_middleware_applies_context():
    state = make_state(context={"name": "Ada"})
    context_middleware(state)
    templating_middleware(state)
    assert state.kwargs["messages"][0]["content"].strip() == "Hello Ada"


def test_cache_lookup_middleware_hits_cached_response():
    cache = AutoCache()
    state = make_state(cache=cache)
    key = make_cache_key(
        messages=state.kwargs["messages"],
        model=state.kwargs["model"],
        response_model=DummyModel,
        mode=state.mode.value,
    )
    store_cached_response(cache, key, DummyModel(value="cached"))

    cache_lookup_middleware(state)

    assert state.short_circuit is True
    assert isinstance(state.result, DummyModel)
    assert state.result.value == "cached"
    assert state.cache_key == key


def test_cache_store_middleware_persists_response():
    cache = AutoCache()
    state = make_state(cache=cache)
    key = make_cache_key(
        messages=state.kwargs["messages"],
        model=state.kwargs["model"],
        response_model=DummyModel,
        mode=state.mode.value,
    )
    state.cache_key = key
    state.result = DummyModel(value="store")

    cache_store_middleware(state)

    assert cache.get(key) is not None


def test_retry_sync_middleware_invokes_retry(monkeypatch):
    called = {}

    def fake_retry_sync(**kwargs):  # type: ignore[no-untyped-def]
        called["params"] = kwargs
        return "ok"

    monkeypatch.setattr("instructor.core.create_pipeline.retry_sync", fake_retry_sync)
    state = make_state()

    retry_sync_middleware(state)

    assert state.result == "ok"
    assert called["params"]["func"] is state.func


@pytest.mark.asyncio
async def test_retry_async_middleware_invokes_retry(monkeypatch):
    called = {}

    async def fake_retry_async(**kwargs):  # type: ignore[no-untyped-def]
        called["params"] = kwargs
        return "async-ok"

    monkeypatch.setattr("instructor.core.create_pipeline.retry_async", fake_retry_async)
    state = make_state()

    await retry_async_middleware(state)

    assert state.result == "async-ok"
    assert called["params"]["func"] is state.func

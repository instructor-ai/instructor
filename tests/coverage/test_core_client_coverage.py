from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any, get_args, get_origin, get_overloads

import pytest
from pydantic import BaseModel, PrivateAttr

from instructor.v2.core import client as core_client
from instructor.v2.core.client import AsyncInstructor, Instructor, Response
from instructor.v2.core.hooks import HookName, Hooks
from instructor.v2.core.mode import Mode


class User(BaseModel):
    name: str
    _raw_response: Any = PrivateAttr(default=None)


MESSAGES = [{"role": "user", "content": "Return a user"}]


def hooks_with_handler(label: str, events: list[str]) -> Hooks:
    hooks = Hooks()
    hooks.on(HookName.COMPLETION_RESPONSE, lambda _response: events.append(label))
    return hooks


def assert_combined_hooks(hooks: Hooks, events: list[str]) -> None:
    hooks.emit_completion_response(object())
    assert events[-2:] == ["client", "call"]


def test_registry_loader_imports_v2_and_tolerates_an_optional_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))
    core_client._ensure_registry_loaded()
    assert imported == ["instructor.v2"]

    def fail_import(name: str) -> None:
        imported.append(name)
        raise ImportError("optional provider is unavailable")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    assert core_client._ensure_registry_loaded() is None
    assert imported == ["instructor.v2", "instructor.v2"]


def test_response_normalizes_input_alias_and_rejects_ambiguous_messages() -> None:
    kwargs: dict[str, Any] = {"input": "hello", "temperature": 0}

    assert Response._normalize_messages(None, kwargs) == [
        {"role": "user", "content": "hello"}
    ]
    assert kwargs == {"temperature": 0}
    assert Response._normalize_messages(MESSAGES, {}) is MESSAGES

    with pytest.raises(TypeError, match="Either 'messages' or 'input'"):
        Response._normalize_messages(None, {})
    with pytest.raises(TypeError, match="Pass only one of 'messages' or 'input'"):
        Response._normalize_messages(MESSAGES, {"input": "duplicate"})


def test_sync_client_forwards_defaults_aliases_and_per_call_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        Mode,
        "warn_mode_functions_deprecation",
        lambda: warnings.append("functions"),
    )
    events: list[str] = []
    client_hooks = hooks_with_handler("client", events)
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []
    underlying = type("UnderlyingClient", (), {"account": "local"})()

    def create(**kwargs: Any) -> User:
        received.append(kwargs)
        return User(name="Ada")

    client = Instructor(
        client=underlying,
        create=create,
        mode=Mode.FUNCTIONS,
        hooks=client_hooks,
        model="default-model",
        temperature=0.1,
    )

    result = client.create(
        response_model=User,
        messages=MESSAGES,
        hooks=call_hooks,
        model="call-model",
    )

    assert result == User(name="Ada")
    assert warnings == ["functions"]
    assert received[0]["model"] == "call-model"
    assert received[0]["temperature"] == 0.1
    assert received[0]["messages"] is MESSAGES
    assert received[0]["response_model"] is User
    assert_combined_hooks(received[0]["hooks"], events)
    assert client.chat.completions.messages is client
    assert client.account == "local"
    assert Instructor.__getattr__(client, "create").__self__ is client


def test_sync_streaming_and_completion_helpers_preserve_models_and_hooks() -> None:
    events: list[str] = []
    client_hooks = hooks_with_handler("client", events)
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []
    raw = object()

    def create(**kwargs: Any) -> Any:
        received.append(kwargs)
        response_model = kwargs["response_model"]
        if kwargs.get("stream"):
            if get_origin(response_model) is Iterable:
                return iter([User(name="Ada"), User(name="Grace")])
            return iter([User.model_construct(name=None), User(name="Ada")])
        result = User(name="Ada")
        result._raw_response = raw
        return result

    client = Instructor(
        client=None,
        create=create,
        hooks=client_hooks,
        model="default-model",
    )

    partial = list(
        client.create_partial(
            response_model=User,
            messages=MESSAGES,
            hooks=call_hooks,
            strict=False,
        )
    )
    iterable = list(
        client.create_iterable(
            response_model=User,
            messages=MESSAGES,
            hooks=call_hooks,
        )
    )
    parsed, completion = client.create_with_completion(
        response_model=User,
        messages=MESSAGES,
        hooks=call_hooks,
    )

    assert partial[0].name is None
    assert partial[1:] == [User(name="Ada")]
    assert received[0]["response_model"]._original_model is User
    assert received[0]["stream"] is True
    assert received[0]["strict"] is False
    assert iterable == [User(name="Ada"), User(name="Grace")]
    assert get_origin(received[1]["response_model"]) is Iterable
    assert get_args(received[1]["response_model"]) == (User,)
    assert received[1]["stream"] is True
    assert parsed.model_dump() == {"name": "Ada"}
    assert completion is raw
    assert all(call["model"] == "default-model" for call in received)
    for call in received:
        assert_combined_hooks(call["hooks"], events)


@pytest.mark.asyncio
async def test_async_client_forwards_defaults_and_dispatches_iterable_response() -> (
    None
):
    events: list[str] = []
    client_hooks = hooks_with_handler("client", events)
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        received.append(kwargs)
        if kwargs.get("stream"):

            async def stream() -> Any:
                yield User(name="Ada")
                yield User(name="Grace")

            return stream()
        return User(name="Ada")

    client = AsyncInstructor(
        client=None,
        create=create,
        hooks=client_hooks,
        model="default-model",
    )

    parsed = await client.create(
        response_model=User,
        messages=MESSAGES,
        hooks=call_hooks,
        temperature=0,
    )
    stream = await client.create(
        response_model=Iterable[User],
        messages=MESSAGES,
        hooks=call_hooks,
    )
    iterable = [item async for item in stream]

    assert parsed.model_dump() == {"name": "Ada"}
    assert iterable == [User(name="Ada"), User(name="Grace")]
    assert received[0]["model"] == "default-model"
    assert received[0]["temperature"] == 0
    assert get_origin(received[1]["response_model"]) is Iterable
    assert get_args(received[1]["response_model"]) == (User,)
    assert received[1]["stream"] is True
    for call in received:
        assert_combined_hooks(call["hooks"], events)


@pytest.mark.asyncio
async def test_async_streaming_and_completion_helpers_preserve_models_and_hooks() -> (
    None
):
    events: list[str] = []
    client_hooks = hooks_with_handler("client", events)
    call_hooks = hooks_with_handler("call", events)
    received: list[dict[str, Any]] = []
    raw = object()

    async def create(**kwargs: Any) -> Any:
        received.append(kwargs)
        if kwargs.get("stream"):

            async def stream() -> Any:
                yield User(name="Ada")

            return stream()
        result = User(name="Ada")
        result._raw_response = raw
        return result

    client = AsyncInstructor(
        client=None,
        create=create,
        hooks=client_hooks,
        model="default-model",
    )

    partial = [
        item
        async for item in client.create_partial(
            response_model=User,
            messages=MESSAGES,
            hooks=call_hooks,
        )
    ]
    iterable = [
        item
        async for item in client.create_iterable(
            response_model=User,
            messages=MESSAGES,
            hooks=call_hooks,
        )
    ]
    parsed, completion = await client.create_with_completion(
        response_model=User,
        messages=MESSAGES,
        hooks=call_hooks,
    )

    assert partial == [User(name="Ada")]
    assert received[0]["response_model"]._original_model is User
    assert received[0]["stream"] is True
    assert iterable == [User(name="Ada")]
    assert get_origin(received[1]["response_model"]) is Iterable
    assert get_args(received[1]["response_model"]) == (User,)
    assert received[1]["stream"] is True
    assert parsed.model_dump() == {"name": "Ada"}
    assert completion is raw
    assert all(call["model"] == "default-model" for call in received)
    for call in received:
        assert_combined_hooks(call["hooks"], events)


def test_openai_compat_wrapper_delegates_and_keeps_its_sync_async_overloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instructor.v2.providers.openai import client as openai_client

    received: list[dict[str, Any]] = []
    expected = object()
    source_client = object()

    monkeypatch.setattr(
        openai_client,
        "from_openai",
        lambda **kwargs: received.append(kwargs) or expected,
    )

    assert (
        core_client.from_openai(
            source_client,
            mode=Mode.JSON,
            model="local-model",
        )
        is expected
    )
    assert received == [
        {"client": source_client, "mode": Mode.JSON, "model": "local-model"}
    ]

    overloads = get_overloads(core_client.from_openai)
    assert len(overloads) == 2
    assert all(overload(source_client) is None for overload in overloads)


@pytest.mark.parametrize("async_client", [True, False, None])
def test_litellm_compat_wrapper_preserves_explicit_and_inferred_client_modes(
    monkeypatch: pytest.MonkeyPatch,
    async_client: bool | None,
) -> None:
    from instructor.v2.providers.litellm import client as litellm_client

    received: list[dict[str, Any]] = []
    expected = object()

    def completion(**_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(
        litellm_client,
        "from_litellm",
        lambda **kwargs: received.append(kwargs) or expected,
    )

    assert (
        core_client.from_litellm(
            completion,
            mode=Mode.JSON,
            async_client=async_client,
            model="local-model",
        )
        is expected
    )
    forwarded = received.pop()
    assert forwarded["completion"] is completion
    assert forwarded["mode"] is Mode.JSON
    assert forwarded["model"] == "local-model"
    if async_client is None:
        assert "async_client" not in forwarded
    else:
        assert forwarded["async_client"] is async_client

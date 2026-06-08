"""Shared contracts for declarative provider client factories."""

from __future__ import annotations

import importlib.util
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest

from instructor import Mode, Provider
from instructor.v2.core import client_factory
from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.registry import mode_registry
from instructor.v2.providers.anthropic import client as anthropic_client
from instructor.v2.providers.cohere import client as cohere_client
from instructor.v2.providers.gemini import client as gemini_client
from instructor.v2.providers.vertexai import client as vertexai_client
from tests.v2.provider_matrix import TEST_PROVIDER_SPECS, ensure_handlers_loaded


def _dependency_missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:
        return True


def test_manifest_exposes_complete_declarative_client_contracts() -> None:
    public = __import__("instructor.v2", fromlist=["__all__"])
    custom_factories = {Provider.LITELLM, Provider.XAI}
    for provider, spec in TEST_PROVIDER_SPECS.items():
        ensure_handlers_loaded(provider)
        assert callable(getattr(public, spec.from_function or "")), provider
        for mode in spec.supported_modes:
            handlers = mode_registry.get_handlers(provider, mode)
            assert all(
                (
                    handlers.request_handler,
                    handlers.reask_handler,
                    handlers.response_parser,
                )
            ), (provider, mode)
        if provider not in custom_factories:
            assert spec.client and spec.client.sync_types and spec.client.create, (
                provider
            )


def test_missing_optional_sdks_raise_shared_client_error() -> None:
    for spec in TEST_PROVIDER_SPECS.values():
        if spec.sdk_module is None or not _dependency_missing(spec.sdk_module):
            continue
        module = __import__(
            spec.client_module or "", fromlist=[spec.from_function or ""]
        )
        with pytest.raises(ClientError, match=spec.missing_sdk_message):
            getattr(module, spec.from_function or "")("not a client")


class _SyncClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        create = lambda **kwargs: self.calls.append(kwargs) or object()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.custom_create = create


class _AsyncClient(_SyncClient):
    pass


def _fake_types(paths: tuple[str, ...], _message: str) -> tuple[type[Any], ...]:
    return (_AsyncClient,) if any("Async" in path for path in paths) else (_SyncClient,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native", "wrapper_type"),
    [(_SyncClient(), Instructor), (_AsyncClient(), AsyncInstructor)],
)
async def test_shared_factory_selects_wrapper_and_preserves_native_stream_flag(
    monkeypatch: pytest.MonkeyPatch,
    native: _SyncClient,
    wrapper_type: type[Instructor] | type[AsyncInstructor],
) -> None:
    monkeypatch.setattr(client_factory, "_resolve_types", _fake_types)
    monkeypatch.setattr(client_factory, "patch_v2", lambda *, func, **_kwargs: func)
    wrapped = client_factory.create_instructor(
        native, provider=Provider.GROQ, mode=Mode.TOOLS
    )
    result = wrapped.create_fn(stream=True)
    if inspect.isawaitable(result):
        await result
    assert isinstance(wrapped, wrapper_type)
    assert native.calls == [{"stream": True}]


def _set_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        child = getattr(root, part, None)
        if child is None:
            child = SimpleNamespace()
            setattr(root, part, child)
        root = child
    setattr(root, parts[-1], value)


_STREAM_CASES = tuple(
    (provider, is_async)
    for provider, spec in PROVIDER_SPECS.items()
    if spec.client is not None
    for is_async, path in (
        (False, spec.client.stream),
        (True, spec.client.async_stream),
    )
    if path is not None
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "is_async"), _STREAM_CASES)
async def test_declared_stream_paths_resolve_and_switch(
    monkeypatch: pytest.MonkeyPatch, provider: Provider, is_async: bool
) -> None:
    contract = PROVIDER_SPECS[provider].client
    assert contract is not None
    calls: list[dict[str, Any]] = []

    async def async_stream(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    native = SimpleNamespace()
    create_path = (
        (contract.async_create or contract.create) if is_async else contract.create
    )
    stream_path = contract.async_stream if is_async else contract.stream
    _set_path(
        native, create_path, async_stream if is_async else lambda **_kwargs: object()
    )
    _set_path(
        native,
        stream_path or "",
        async_stream if is_async else lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(client_factory, "patch_v2", lambda *, func, **_kwargs: func)
    wrapped = client_factory.create_instructor(
        native,
        provider=provider,
        mode=PROVIDER_SPECS[provider].supported_modes[0],
        model="default-model",
        use_async=is_async,
        sync_types=(SimpleNamespace,),
        async_types=(SimpleNamespace,),
    )
    result = wrapped.create_fn(stream=True, model=None, value=1)
    if inspect.isawaitable(result):
        await result
    expected = {
        "model": "default-model" if contract.falsey_model_fallback else None,
        "value": 1,
    }
    assert calls == [expected]


def _missing_types(*_args: Any) -> tuple[type[Any], ...]:
    raise ClientError("missing dependency")


def _unexpected_types(*_args: Any) -> tuple[type[Any], ...]:
    raise AssertionError("mode validation must run first")


@pytest.mark.parametrize(
    ("provider", "resolver", "error"),
    [
        (Provider.GROQ, _missing_types, ClientError),
        (Provider.GROQ, _fake_types, ModeError),
        (Provider.GEMINI, _unexpected_types, ModeError),
        (Provider.GENAI, _fake_types, ClientError),
    ],
)
def test_shared_factory_preserves_validation_precedence(
    monkeypatch: pytest.MonkeyPatch, provider: Provider, resolver: Any, error: Any
) -> None:
    monkeypatch.setattr(client_factory, "_resolve_types", resolver)
    with pytest.raises(error):
        client_factory.create_instructor(
            object(), provider=provider, mode=Mode.RESPONSES_TOOLS
        )


@pytest.mark.parametrize(
    ("module", "sdk_attribute", "factory"),
    [
        (gemini_client, "genai", gemini_client.from_gemini),
        (vertexai_client, "gm", vertexai_client.from_vertexai),
    ],
)
def test_mode_first_adapters_validate_before_missing_sdk(
    monkeypatch: pytest.MonkeyPatch, module: Any, sdk_attribute: str, factory: Any
) -> None:
    monkeypatch.setattr(module, sdk_attribute, None)
    with pytest.raises(ModeError):
        factory(object(), mode=Mode.RESPONSES_TOOLS)


def _capture_factory(monkeypatch: pytest.MonkeyPatch, module: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        module,
        "create_instructor",
        lambda _client, **kwargs: captured.update(kwargs) or object(),
    )
    return captured


def test_anthropic_beta_declares_beta_method_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_type, async_type = type("Sync", (), {}), type("Async", (), {})
    monkeypatch.setattr(
        anthropic_client,
        "anthropic",
        SimpleNamespace(
            Anthropic=sync_type,
            AnthropicBedrock=sync_type,
            AnthropicVertex=sync_type,
            AsyncAnthropic=async_type,
            AsyncAnthropicBedrock=async_type,
            AsyncAnthropicVertex=async_type,
        ),
    )
    captured = _capture_factory(monkeypatch, anthropic_client)
    cast(Any, anthropic_client.from_anthropic)(sync_type(), beta=True)
    assert captured["create_path"] == "beta.messages.create"
    assert captured["async_create_path"] == "beta.messages.create"


def test_cohere_v2_declares_version_and_client_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1, v2, async_v1, async_v2 = (type(name, (), {}) for name in "V1 V2 A1 A2".split())
    monkeypatch.setattr(
        cohere_client,
        "cohere",
        SimpleNamespace(
            Client=v1, ClientV2=v2, AsyncClient=async_v1, AsyncClientV2=async_v2
        ),
    )
    captured = _capture_factory(monkeypatch, cohere_client)
    cast(Any, cohere_client.from_cohere)(v2())
    assert captured["_cohere_client_version"] == "v2"
    assert captured["sync_types"] == (v1, v2)
    assert captured["async_types"] == (async_v1, async_v2)

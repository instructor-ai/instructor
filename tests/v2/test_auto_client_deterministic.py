from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from instructor import Mode
from instructor.v2 import auto_client
from instructor.v2.core.errors import ConfigurationError


class DummyCache:
    pass


def test_from_provider_requires_provider_prefix() -> None:
    with pytest.raises(ConfigurationError, match="Model string must be in format"):
        auto_client.from_provider("gpt-5")


def test_from_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported provider: mystery"):
        auto_client.from_provider("mystery/model")


def test_from_provider_passes_cache_and_api_key_to_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    cache = DummyCache()

    def fake_builder(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "client"

    monkeypatch.setitem(auto_client._PROVIDER_BUILDERS, "openai", fake_builder)

    result = auto_client.from_provider(
        "openai/gpt-5-nano",
        cache=cache,
        api_key="secret",
        mode=Mode.JSON_SCHEMA,
        timeout=30,
    )

    assert result == "client"
    assert captured["provider"] == "openai"
    assert captured["model_name"] == "gpt-5-nano"
    assert captured["api_key"] == "secret"
    assert captured["mode"] == Mode.JSON_SCHEMA
    assert captured["kwargs"]["cache"] is cache
    assert captured["kwargs"]["timeout"] == 30
    assert "api_key" not in captured["kwargs"]


def test_build_openai_compatible_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYSCALE_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="ANYSCALE_API_KEY is not set"):
        auto_client._build_openai_compatible(
            provider="anyscale",
            model_name="llama",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={},
            provider_info={"provider": "anyscale", "operation": "initialize"},
            env_var="ANYSCALE_API_KEY",
            default_base_url="https://api.endpoints.anyscale.com/v1",
            factory_name="from_anyscale",
        )


def test_build_openai_does_not_mask_runtime_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_module = ModuleType("openai")
    httpx_module = ModuleType("httpx")

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise ImportError("Using SOCKS proxy, but socksio is not installed.")

    openai_module.OpenAI = FakeClient  # type: ignore[attr-defined]
    openai_module.AsyncOpenAI = FakeClient  # type: ignore[attr-defined]
    openai_module.DEFAULT_MAX_RETRIES = 2  # type: ignore[attr-defined]
    openai_module.NotGiven = object  # type: ignore[attr-defined]
    openai_module.Timeout = float  # type: ignore[attr-defined]
    openai_module.not_given = object()  # type: ignore[attr-defined]
    httpx_module.Client = object  # type: ignore[attr-defined]
    httpx_module.AsyncClient = object  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "openai", openai_module)
    monkeypatch.setitem(__import__("sys").modules, "httpx", httpx_module)

    with pytest.raises(ImportError, match="socksio"):
        auto_client._build_openai(
            provider="openai",
            model_name="gpt-5",
            async_client=False,
            mode=Mode.TOOLS,
            api_key="test-key",
            kwargs={},
            provider_info={"provider": "openai", "operation": "initialize"},
        )


def test_build_databricks_normalizes_base_url_and_forwards_client_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "db-token")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.databricks.com")

    openai_module = ModuleType("openai")
    seen: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            seen["client_kwargs"] = kwargs

    openai_module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    openai_module.AsyncOpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "openai", openai_module)

    import instructor

    def fake_from_openai(_client: Any, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = _client
        seen["factory_kwargs"] = kwargs
        return {"client": _client, "kwargs": kwargs}

    monkeypatch.setattr(instructor, "from_openai", fake_from_openai)

    result = auto_client._build_databricks(
        provider="databricks",
        model_name="meta-llama",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={"timeout": 10, "custom": "value"},
        provider_info={"provider": "databricks", "operation": "initialize"},
    )

    assert result["kwargs"]["model"] == "meta-llama"
    assert result["kwargs"]["mode"] == Mode.TOOLS
    assert result["kwargs"]["custom"] == "value"
    assert seen["client_kwargs"]["api_key"] == "db-token"
    assert (
        seen["client_kwargs"]["base_url"]
        == "https://workspace.databricks.com/serving-endpoints"
    )
    assert seen["client_kwargs"]["timeout"] == 10


def test_build_bedrock_chooses_default_mode_from_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3_module = ModuleType("boto3")
    boto3_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_client(service_name: str, **kwargs: Any) -> object:
        boto3_calls.append((service_name, kwargs))
        return object()

    boto3_module.client = fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "boto3", boto3_module)

    import instructor.v2.providers.bedrock.client as bedrock_client

    calls: list[dict[str, Any]] = []

    def fake_from_bedrock(_client: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(bedrock_client, "from_bedrock", fake_from_bedrock)

    auto_client._build_bedrock(
        provider="bedrock",
        model_name="anthropic.claude-3-7-sonnet",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info={"provider": "bedrock", "operation": "initialize"},
    )
    auto_client._build_bedrock(
        provider="bedrock",
        model_name="amazon.titan-text",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info={"provider": "bedrock", "operation": "initialize"},
    )

    assert boto3_calls[0][0] == "bedrock-runtime"
    assert calls[0]["mode"] == Mode.TOOLS
    assert calls[1]["mode"] == Mode.MD_JSON


def test_build_ollama_uses_tool_mode_only_for_tool_capable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_module = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    openai_module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    openai_module.AsyncOpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "openai", openai_module)

    import instructor.v2.providers.openai.client as openai_client_module

    calls: list[dict[str, Any]] = []

    def fake_from_openai(_client: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(openai_client_module, "from_openai", fake_from_openai)

    auto_client._build_ollama(
        provider="ollama",
        model_name="llama3.1:8b",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info={"provider": "ollama", "operation": "initialize"},
    )
    auto_client._build_ollama(
        provider="ollama",
        model_name="phi4-mini",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info={"provider": "ollama", "operation": "initialize"},
    )

    assert calls[0]["mode"] == Mode.TOOLS
    assert calls[1]["mode"] == Mode.JSON

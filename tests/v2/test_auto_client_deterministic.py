from __future__ import annotations

from types import ModuleType
from typing import Any, cast

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

    result = auto_client.from_provider(  # ty: ignore[no-matching-overload]
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

    setattr(openai_module, "OpenAI", FakeClient)  # noqa: B010
    setattr(openai_module, "AsyncOpenAI", FakeClient)  # noqa: B010
    setattr(openai_module, "DEFAULT_MAX_RETRIES", 2)  # noqa: B010
    setattr(openai_module, "NotGiven", object)  # noqa: B010
    setattr(openai_module, "Timeout", float)  # noqa: B010
    setattr(openai_module, "not_given", object())  # noqa: B010
    setattr(httpx_module, "Client", object)  # noqa: B010
    setattr(httpx_module, "AsyncClient", object)  # noqa: B010

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

    setattr(openai_module, "OpenAI", FakeOpenAI)  # noqa: B010
    setattr(openai_module, "AsyncOpenAI", FakeOpenAI)  # noqa: B010
    monkeypatch.setitem(__import__("sys").modules, "openai", openai_module)

    import instructor

    def fake_from_openai(_client: Any, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = _client
        seen["factory_kwargs"] = kwargs
        return {"client": _client, "kwargs": kwargs}

    monkeypatch.setattr(instructor, "from_openai", fake_from_openai)

    result = cast(
        dict[str, Any],
        auto_client._build_databricks(
            provider="databricks",
            model_name="meta-llama",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={"timeout": 10, "custom": "value"},
            provider_info={"provider": "databricks", "operation": "initialize"},
        ),
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

    setattr(boto3_module, "client", fake_client)  # noqa: B010
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


def _install_fake_bedrock_deps(
    monkeypatch: pytest.MonkeyPatch, *, with_token_classes: bool = True
) -> dict[str, Any]:
    """Install fake ``boto3`` / ``botocore.tokens`` modules and record what the
    Bedrock client factory does with them.

    Returns a holder dict capturing plain ``boto3.client`` calls, session-scoped
    ``Session().client`` calls, and any component registered on the session.
    """
    import sys

    holder: dict[str, Any] = {
        "plain_client_calls": [],
        "session_client_calls": [],
        "session": None,
    }

    boto3_module = ModuleType("boto3")

    def fake_client(service_name: str, **kwargs: Any) -> object:
        holder["plain_client_calls"].append((service_name, kwargs))
        return object()

    setattr(boto3_module, "client", fake_client)  # noqa: B010

    class FakeBotocoreSession:
        def __init__(self) -> None:
            self.registered: dict[str, Any] = {}

        def register_component(self, name: str, value: Any) -> None:
            self.registered[name] = value

    class FakeSession:
        def __init__(self) -> None:
            self._session = FakeBotocoreSession()
            holder["session"] = self

        def client(self, service_name: str, **kwargs: Any) -> object:
            holder["session_client_calls"].append((service_name, kwargs))
            return object()

    session_module = ModuleType("boto3.session")
    setattr(session_module, "Session", FakeSession)  # noqa: B010
    setattr(boto3_module, "session", session_module)  # noqa: B010
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "boto3.session", session_module)

    botocore_module = ModuleType("botocore")
    tokens_module = ModuleType("botocore.tokens")
    if with_token_classes:

        class FakeScopedEnvTokenProvider:
            def __init__(self, session: Any, environ: Any = None) -> None:
                self.session = session
                self.environ = environ

        class FakeSSOTokenProvider:
            def __init__(self, session: Any) -> None:
                self.session = session

        class FakeTokenProviderChain:
            def __init__(self, providers: Any) -> None:
                self.providers = providers

        setattr(tokens_module, "ScopedEnvTokenProvider", FakeScopedEnvTokenProvider)  # noqa: B010
        setattr(tokens_module, "SSOTokenProvider", FakeSSOTokenProvider)  # noqa: B010
        setattr(tokens_module, "TokenProviderChain", FakeTokenProviderChain)  # noqa: B010
    setattr(botocore_module, "tokens", tokens_module)  # noqa: B010
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.tokens", tokens_module)

    import instructor.v2.providers.bedrock.client as bedrock_client

    monkeypatch.setattr(
        bedrock_client, "from_bedrock", lambda _client, **kwargs: kwargs
    )

    return holder


def test_build_bedrock_api_key_scopes_bearer_token_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    holder = _install_fake_bedrock_deps(monkeypatch)

    # Ensure the process-wide variable is absent (restored on teardown).
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "sentinel")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")

    auto_client._build_bedrock(
        provider="bedrock",
        model_name="anthropic.claude-3-7-sonnet",
        async_client=False,
        mode=None,
        api_key="bedrock-api-key",
        kwargs={},
        provider_info={"provider": "bedrock", "operation": "initialize"},
    )

    # The client is built from a dedicated session, not the module-level factory.
    assert holder["session_client_calls"][0][0] == "bedrock-runtime"
    assert holder["plain_client_calls"] == []

    # The bearer token is scoped to that session's token provider...
    chain = holder["session"]._session.registered["token_provider"]
    scoped_provider = chain.providers[0]
    assert scoped_provider.environ == {"AWS_BEARER_TOKEN_BEDROCK": "bedrock-api-key"}

    # ...and never leaks into the process environment.
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ


def test_build_bedrock_without_api_key_uses_plain_client_and_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    holder = _install_fake_bedrock_deps(monkeypatch)

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "sentinel")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")

    auto_client._build_bedrock(
        provider="bedrock",
        model_name="anthropic.claude-3-7-sonnet",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info={"provider": "bedrock", "operation": "initialize"},
    )

    # No api_key -> plain boto3.client, no scoped session, no env mutation.
    assert holder["plain_client_calls"][0][0] == "bedrock-runtime"
    assert holder["session"] is None
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ


def test_build_bedrock_api_key_requires_bearer_token_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # botocore too old to expose the bearer-token providers -> clear error,
    # never a silent fall-through to SigV4 that ignores the key.
    _install_fake_bedrock_deps(monkeypatch, with_token_classes=False)

    with pytest.raises(ConfigurationError, match="botocore >= 1.39.0"):
        auto_client._build_bedrock(
            provider="bedrock",
            model_name="anthropic.claude-3-7-sonnet",
            async_client=False,
            mode=None,
            api_key="bedrock-api-key",
            kwargs={},
            provider_info={"provider": "bedrock", "operation": "initialize"},
        )


def test_build_ollama_uses_tool_mode_only_for_tool_capable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_module = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    setattr(openai_module, "OpenAI", FakeOpenAI)  # noqa: B010
    setattr(openai_module, "AsyncOpenAI", FakeOpenAI)  # noqa: B010
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

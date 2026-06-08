from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import pytest

from instructor import Mode
from instructor.v2 import auto_client
from instructor.v2.core.errors import ConfigurationError


class DummyCache:
    pass


def _module(path: str) -> ModuleType:
    module = ModuleType(path)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def test_from_provider_requires_provider_prefix() -> None:
    with pytest.raises(ConfigurationError, match="Model string must be in format"):
        auto_client.from_provider("gpt-5")


def test_from_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported provider: mystery"):
        auto_client.from_provider("mystery/model")


def test_provider_builders_are_derived_from_supported_aliases() -> None:
    routed_aliases = {
        alias
        for alias, provider in auto_client.ALIAS_TO_PROVIDER.items()
        if auto_client.PROVIDER_SPECS[provider].model_builder_module is not None
    }
    assert routed_aliases == set(auto_client.supported_providers)


def test_from_provider_passes_cache_and_api_key_to_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    cache = DummyCache()

    def fake_builder(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "client"

    builder_module = cast(Any, ModuleType("test_builder"))
    builder_module.build_from_model = fake_builder
    monkeypatch.setattr(
        auto_client.importlib,
        "import_module",
        lambda _path: builder_module,
    )

    result = auto_client.from_provider(  # ty: ignore[no-matching-overload]
        "openai/gpt-5-nano",
        cache=cache,
        api_key="secret",
        mode=Mode.JSON_SCHEMA,
        timeout=30,
    )

    assert result == "client"
    assert captured["provider"] == auto_client.ALIAS_TO_PROVIDER["openai"]
    assert captured["model_name"] == "gpt-5-nano"
    assert captured["api_key"] == "secret"
    assert captured["mode"] == Mode.JSON_SCHEMA
    assert captured["kwargs"]["cache"] is cache
    assert captured["kwargs"]["timeout"] == 30
    assert "api_key" not in captured["kwargs"]


@pytest.mark.parametrize(
    (
        "_provider",
        "sdk_modules",
        "factory_module",
        "factory_name",
        "expected_default",
    ),
    [
        (
            "cohere",
            {"cohere": ("ClientV2", "AsyncClientV2")},
            "instructor.v2.providers.cohere.client",
            "from_cohere",
            Mode.TOOLS,
        ),
        (
            "mistral",
            {"mistralai": ("Mistral",)},
            "instructor.v2.providers.mistral.client",
            "from_mistral",
            Mode.TOOLS,
        ),
        (
            "groq",
            {"groq": ("Groq", "AsyncGroq")},
            "instructor.v2.providers.groq.client",
            "from_groq",
            Mode.TOOLS,
        ),
        (
            "writer",
            {"writerai": ("Writer", "AsyncWriter")},
            "instructor.v2.providers.writer.client",
            "from_writer",
            Mode.TOOLS,
        ),
        (
            "cerebras",
            {
                "cerebras": (),
                "cerebras.cloud": (),
                "cerebras.cloud.sdk": ("Cerebras", "AsyncCerebras"),
            },
            "instructor.v2.providers.cerebras.client",
            "from_cerebras",
            Mode.TOOLS,
        ),
        (
            "fireworks",
            {
                "fireworks": (),
                "fireworks.client": ("Fireworks", "AsyncFireworks"),
            },
            "instructor.v2.providers.fireworks.client",
            "from_fireworks",
            Mode.TOOLS,
        ),
    ],
)
def test_builders_forward_requested_mode(
    monkeypatch: pytest.MonkeyPatch,
    _provider: str,
    sdk_modules: dict[str, tuple[str, ...]],
    factory_module: str,
    factory_name: str,
    expected_default: Mode,
) -> None:
    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    factory_module_obj = __import__(factory_module, fromlist=[factory_name])
    for module_path, class_names in sdk_modules.items():
        module = _module(module_path)
        for class_name in class_names:
            setattr(module, class_name, FakeClient)
        monkeypatch.setitem(__import__("sys").modules, module_path, module)

    factory_calls: list[dict[str, Any]] = []

    def fake_factory(_client: Any, **kwargs: Any) -> dict[str, Any]:
        factory_calls.append(kwargs)
        return kwargs

    if _provider == "cohere":
        monkeypatch.setattr(
            factory_module_obj, "cohere", __import__("sys").modules["cohere"]
        )
    elif _provider == "mistral":
        monkeypatch.setattr(factory_module_obj, "Mistral", FakeClient)
    elif _provider == "groq":
        monkeypatch.setattr(
            factory_module_obj, "groq", __import__("sys").modules["groq"]
        )
    elif _provider == "writer":
        monkeypatch.setattr(factory_module_obj, "Writer", FakeClient)
        monkeypatch.setattr(factory_module_obj, "AsyncWriter", FakeClient)
    elif _provider == "cerebras":
        monkeypatch.setattr(factory_module_obj, "Cerebras", FakeClient)
        monkeypatch.setattr(factory_module_obj, "AsyncCerebras", FakeClient)
    elif _provider == "fireworks":
        monkeypatch.setattr(factory_module_obj, "Fireworks", FakeClient)
        monkeypatch.setattr(factory_module_obj, "AsyncFireworks", FakeClient)
    monkeypatch.setattr(factory_module_obj, factory_name, fake_factory)

    builder = cast(Any, factory_module_obj).build_from_model
    builder(
        provider=auto_client.ALIAS_TO_PROVIDER[_provider],
        model_name="test-model",
        async_client=False,
        mode=Mode.MD_JSON,
        api_key="test-key",
        kwargs={},
    )
    builder(
        provider=auto_client.ALIAS_TO_PROVIDER[_provider],
        model_name="test-model",
        async_client=False,
        mode=None,
        api_key="test-key",
        kwargs={},
    )

    assert factory_calls[0]["mode"] == Mode.MD_JSON
    assert factory_calls[1]["mode"] == expected_default


def test_build_openai_compatible_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANYSCALE_API_KEY", raising=False)

    from instructor.v2.providers.openai import client as openai_client

    with pytest.raises(ConfigurationError, match="ANYSCALE_API_KEY is not set"):
        openai_client.build_from_model(
            provider=auto_client.ALIAS_TO_PROVIDER["anyscale"],
            model_name="llama",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={},
        )


def test_build_openai_does_not_mask_runtime_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_module = cast(Any, ModuleType("openai"))

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise ImportError("Using SOCKS proxy, but socksio is not installed.")

    openai_module.OpenAI = FakeClient
    openai_module.AsyncOpenAI = FakeClient
    openai_module.DEFAULT_MAX_RETRIES = 2
    openai_module.NotGiven = object
    openai_module.Timeout = float
    openai_module.not_given = object()

    from instructor.v2.providers.openai import client as openai_client

    monkeypatch.setattr(openai_client, "openai", openai_module)

    with pytest.raises(ImportError, match="socksio"):
        openai_client.build_from_model(
            provider=auto_client.ALIAS_TO_PROVIDER["openai"],
            model_name="gpt-5",
            async_client=False,
            mode=Mode.TOOLS,
            api_key="test-key",
            kwargs={},
        )


@pytest.mark.parametrize("async_client", [False, True])
def test_openai_builder_restores_default_max_retries_for_none(
    monkeypatch: pytest.MonkeyPatch,
    async_client: bool,
) -> None:
    openai_module = cast(Any, ModuleType("openai"))
    seen: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            seen["client_kwargs"] = kwargs

    openai_module.OpenAI = FakeClient
    openai_module.AsyncOpenAI = FakeClient
    openai_module.DEFAULT_MAX_RETRIES = 7

    from instructor.v2.providers.openai import client as openai_client

    monkeypatch.setattr(openai_client, "openai", openai_module)

    openai_client._openai_client(
        async_client=async_client,
        api_key="test-key",
        base_url=None,
        kwargs={"max_retries": None},
    )

    assert seen["client_kwargs"]["max_retries"] == 7


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("compatible", [False, True], ids=["openai", "compatible"])
def test_openai_builders_keep_app_info_for_instructor_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    async_client: bool,
    compatible: bool,
) -> None:
    openai_module = cast(Any, ModuleType("openai"))
    seen: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            assert "app_info" not in kwargs
            seen["client_kwargs"] = kwargs

    openai_module.OpenAI = FakeClient
    openai_module.AsyncOpenAI = FakeClient

    from instructor.v2.providers.openai import client as openai_client

    monkeypatch.setattr(openai_client, "openai", openai_module)

    def fake_factory(_client: Any, **kwargs: Any) -> dict[str, Any]:
        seen["factory_kwargs"] = kwargs
        return kwargs

    if compatible:
        builder = openai_client.compatible_model_builder(
            cast(Any, fake_factory),
            env_var="TEST_API_KEY",
            base_url="https://example.com/v1",
        )
        provider = auto_client.ALIAS_TO_PROVIDER["anyscale"]
    else:
        monkeypatch.setattr(openai_client, "from_openai", fake_factory)
        builder = openai_client.build_from_model
        provider = auto_client.ALIAS_TO_PROVIDER["openai"]

    builder(
        provider=provider,
        model_name="test-model",
        async_client=async_client,
        mode=Mode.TOOLS,
        api_key="test-key",
        kwargs={"app_info": {"name": "instructor"}},
    )

    assert seen["factory_kwargs"]["app_info"] == {"name": "instructor"}


def test_build_databricks_normalizes_base_url_and_forwards_client_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "db-token")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.databricks.com")

    openai_module = cast(Any, ModuleType("openai"))
    seen: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            seen["client_kwargs"] = kwargs

    openai_module.OpenAI = FakeOpenAI
    openai_module.AsyncOpenAI = FakeOpenAI
    from instructor.v2.providers.openai import client as openai_client

    monkeypatch.setattr(openai_client, "openai", openai_module)

    def fake_from_databricks(_client: Any, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = _client
        seen["factory_kwargs"] = kwargs
        return {"client": _client, "kwargs": kwargs}

    monkeypatch.setattr(openai_client, "from_databricks", fake_from_databricks)

    result = cast(
        dict[str, Any],
        openai_client.build_from_model(
            provider=auto_client.ALIAS_TO_PROVIDER["databricks"],
            model_name="meta-llama",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={"timeout": 10, "custom": "value"},
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
    boto3_module = cast(Any, ModuleType("boto3"))
    boto3_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_client(service_name: str, **kwargs: Any) -> object:
        boto3_calls.append((service_name, kwargs))
        return object()

    boto3_module.client = fake_client
    monkeypatch.setitem(__import__("sys").modules, "boto3", boto3_module)

    import instructor.v2.providers.bedrock.client as bedrock_client

    calls: list[dict[str, Any]] = []

    def fake_from_bedrock(_client: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(bedrock_client, "from_bedrock", fake_from_bedrock)

    bedrock_client.build_from_model(
        provider=auto_client.ALIAS_TO_PROVIDER["bedrock"],
        model_name="anthropic.claude-3-7-sonnet",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
    )
    bedrock_client.build_from_model(
        provider=auto_client.ALIAS_TO_PROVIDER["bedrock"],
        model_name="amazon.titan-text",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
    )

    assert boto3_calls[0][0] == "bedrock-runtime"
    assert calls[0]["mode"] == Mode.TOOLS
    assert calls[0]["model"] == "anthropic.claude-3-7-sonnet"
    assert calls[1]["mode"] == Mode.MD_JSON
    assert calls[1]["model"] == "amazon.titan-text"


def test_build_ollama_uses_tool_mode_only_for_tool_capable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_module = cast(Any, ModuleType("openai"))

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    openai_module.OpenAI = FakeOpenAI
    openai_module.AsyncOpenAI = FakeOpenAI
    import instructor.v2.providers.openai.client as openai_client_module

    monkeypatch.setattr(openai_client_module, "openai", openai_module)
    calls: list[dict[str, Any]] = []
    client_kwargs: list[dict[str, Any]] = []

    class CapturingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            client_kwargs.append(kwargs)

    openai_module.OpenAI = CapturingOpenAI
    openai_module.AsyncOpenAI = CapturingOpenAI

    def fake_from_openai(_client: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(openai_client_module, "from_openai", fake_from_openai)

    openai_client_module.build_from_model(
        provider=auto_client.ALIAS_TO_PROVIDER["ollama"],
        model_name="llama3.1:8b",
        async_client=False,
        mode=None,
        api_key="given-key",
        kwargs={},
    )
    openai_client_module.build_from_model(
        provider=auto_client.ALIAS_TO_PROVIDER["ollama"],
        model_name="phi4-mini",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
    )

    assert calls[0]["mode"] == Mode.TOOLS
    assert calls[1]["mode"] == Mode.JSON
    assert client_kwargs[0]["api_key"] == "given-key"
    assert client_kwargs[1]["api_key"] == "ollama"

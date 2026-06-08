"""v2 OpenAI client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.openai import handlers  # noqa: F401


def map_chat_completion_to_response(messages, client, *args, **kwargs) -> Any:
    return client.responses.create(*args, input=messages, **kwargs)


async def async_map_chat_completion_to_response(
    messages, client, *args, **kwargs
) -> Any:
    return await client.responses.create(*args, input=messages, **kwargs)


def _from_openai_compat(
    client: openai.OpenAI | openai.AsyncOpenAI,
    provider: Provider,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.registry import mode_registry, normalize_mode

    normalized_mode = (
        Mode.RESPONSES_TOOLS
        if mode == Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
        else normalize_mode(provider, mode)
    )
    if not mode_registry.is_registered(provider, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(provider)
        raise ModeError(
            mode=str(mode.value),
            provider=provider.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    valid_client_types = (
        openai.OpenAI,
        openai.AsyncOpenAI,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
        create = (
            partial(map_chat_completion_to_response, client=client)
            if isinstance(client, openai.OpenAI)
            else partial(async_map_chat_completion_to_response, client=client)
        )
    else:
        create = client.chat.completions.create
    patched_create = patch_v2(
        func=create,
        provider=provider,
        mode=normalized_mode,
        default_model=model,
    )

    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=patched_create,
            provider=provider,
            mode=normalized_mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=provider,
        mode=normalized_mode,
        **kwargs,
    )


@overload
def from_openai(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an OpenAI client using v2 registry.

    Args:
        client: An instance of OpenAI client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for OpenAI
        ClientError: If client is not a valid OpenAI client instance

    Examples:
        >>> import openai
        >>> from instructor import Mode
        >>> from instructor.v2.providers.openai import from_openai
        >>>
        >>> client = openai.OpenAI()
        >>> instructor_client = from_openai(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use JSON_SCHEMA mode for structured outputs
        >>> instructor_client = from_openai(client, mode=Mode.JSON_SCHEMA)
    """
    return _from_openai_compat(
        client=client,
        provider=Provider.OPENAI,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_anyscale(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_anyscale(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_anyscale(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_anyscale(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_anyscale(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Anyscale.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_anyscale
       >>> client = from_anyscale("Mixtral-8x7B-Instruct-v0.1", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.endpoints.anyscale.com/v1")
       >>> instructor_client = from_anyscale(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Anyscale
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"anyscale/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.ANYSCALE,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_together(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_together(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_together(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_together(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_together(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Together AI.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_together
       >>> client = from_together("Mixtral-8x7B-Instruct-v0.1", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.together.xyz/v1")
       >>> instructor_client = from_together(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Together AI
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"together/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.TOGETHER,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_databricks(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_databricks(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_databricks(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_databricks(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_databricks(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Databricks.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_databricks
       >>> client = from_databricks("dbrx-instruct", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://workspace.cloud.databricks.com/serving-endpoints")
       >>> instructor_client = from_databricks(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Databricks
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"databricks/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.DATABRICKS,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_deepseek(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_deepseek(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_deepseek(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_deepseek(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_deepseek(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for DeepSeek.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_deepseek
       >>> client = from_deepseek("deepseek-chat", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.deepseek.com")
       >>> instructor_client = from_deepseek(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for DeepSeek
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"deepseek/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.DEEPSEEK,
        mode=mode,
        model=model,
        **kwargs,
    )


def _required_api_key(
    api_key: str | None, env_var: str, *, value_name: str = "api-key"
) -> str:
    from instructor.v2.core.errors import ConfigurationError

    resolved = api_key or os.environ.get(env_var)
    if resolved:
        return resolved
    raise ConfigurationError(
        f"{env_var} is not set. Set it with `export {env_var}=<your-{value_name}>` "
        f"or pass it as kwarg api_key=<your-{value_name}>"
    )


def _openai_client(
    *,
    async_client: bool,
    api_key: str | None,
    base_url: str | None,
    kwargs: dict[str, Any],
) -> openai.OpenAI | openai.AsyncOpenAI:
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    for key in (
        "organization",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
        "_strict_response_validation",
    ):
        if key in kwargs:
            value = kwargs.pop(key)
            if key == "max_retries" and value is None:
                value = openai.DEFAULT_MAX_RETRIES
            client_kwargs[key] = value
    factory = openai.AsyncOpenAI if async_client else openai.OpenAI
    return factory(**client_kwargs)


def compatible_model_builder(
    factory: Callable[..., Instructor | AsyncInstructor],
    *,
    env_var: str,
    base_url: str,
    default_mode: Mode = Mode.TOOLS,
) -> Callable[..., Instructor | AsyncInstructor]:
    """Create a lazy model-string builder for identical OpenAI wire clients."""

    def build_from_model(
        *,
        provider: Provider,  # noqa: ARG001
        model_name: str,
        async_client: bool,
        mode: Mode | None,
        api_key: str | None,
        kwargs: dict[str, Any],
    ) -> Instructor | AsyncInstructor:
        client = _openai_client(
            async_client=async_client,
            api_key=_required_api_key(api_key, env_var),
            base_url=kwargs.pop("base_url", base_url),
            kwargs=kwargs,
        )
        return factory(client, model=model_name, mode=mode or default_mode, **kwargs)

    return build_from_model


_COMPAT_BUILDERS = {
    Provider.ANYSCALE: compatible_model_builder(
        from_anyscale,
        env_var="ANYSCALE_API_KEY",
        base_url="https://api.endpoints.anyscale.com/v1",
    ),
    Provider.TOGETHER: compatible_model_builder(
        from_together,
        env_var="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
    ),
    Provider.DEEPSEEK: compatible_model_builder(
        from_deepseek,
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
}


def build_from_model(
    *,
    provider: Provider,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
) -> Instructor | AsyncInstructor:
    """Build a model-string client for OpenAI and OpenAI-compatible providers."""
    selected_mode = mode or Mode.TOOLS
    if provider is Provider.OPENAI:
        client = _openai_client(
            async_client=async_client,
            api_key=api_key,
            base_url=kwargs.pop("base_url", None),
            kwargs=kwargs,
        )
        return from_openai(client, model=model_name, mode=selected_mode, **kwargs)

    if provider is Provider.AZURE_OPENAI:
        from instructor.v2.core.errors import ConfigurationError

        azure_key = _required_api_key(api_key, "AZURE_OPENAI_API_KEY")
        endpoint = kwargs.pop("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT"))
        if not endpoint:
            raise ConfigurationError(
                "AZURE_OPENAI_ENDPOINT is not set. Set it with "
                "`export AZURE_OPENAI_ENDPOINT=<your-endpoint>` or pass it as "
                "kwarg azure_endpoint=<your-endpoint>"
            )
        factory = openai.AsyncAzureOpenAI if async_client else openai.AzureOpenAI
        client = factory(
            api_key=azure_key,
            api_version=kwargs.pop("api_version", "2024-02-01"),
            azure_endpoint=endpoint,
        )
        return from_openai(client, model=model_name, mode=selected_mode, **kwargs)

    if provider is Provider.DATABRICKS:
        from instructor.v2.core.errors import ConfigurationError

        token = (
            api_key
            or os.environ.get("DATABRICKS_TOKEN")
            or os.environ.get("DATABRICKS_API_KEY")
        )
        if not token:
            raise ConfigurationError(
                "DATABRICKS_TOKEN is not set. Set it with "
                "`export DATABRICKS_TOKEN=<your-token>` or "
                "`export DATABRICKS_API_KEY=<your-token>` or pass it as kwarg "
                "`api_key=<your-token>`."
            )
        base_url = (
            kwargs.pop("base_url", None)
            or os.environ.get("DATABRICKS_BASE_URL")
            or os.environ.get("DATABRICKS_HOST")
            or os.environ.get("DATABRICKS_WORKSPACE_URL")
        )
        if not base_url:
            raise ConfigurationError(
                "DATABRICKS_HOST is not set. Set it with "
                "`export DATABRICKS_HOST=<your-workspace-url>` or "
                "`export DATABRICKS_WORKSPACE_URL=<your-workspace-url>` or pass "
                "`base_url=<your-workspace-url>`."
            )
        normalized_url = str(base_url).rstrip("/")
        if not normalized_url.endswith("/serving-endpoints"):
            normalized_url += "/serving-endpoints"
        client = _openai_client(
            async_client=async_client,
            api_key=token,
            base_url=normalized_url,
            kwargs=kwargs,
        )
        return from_databricks(client, model=model_name, mode=selected_mode, **kwargs)

    if provider is Provider.OLLAMA:
        base_url = kwargs.pop("base_url", "http://localhost:11434/v1")
        client = _openai_client(
            async_client=async_client,
            api_key=api_key or "ollama",
            base_url=base_url,
            kwargs=kwargs,
        )
        tool_models = {
            "llama3.1",
            "llama3.2",
            "llama4",
            "mistral-nemo",
            "firefunction-v2",
            "command-a",
            "command-r",
            "command-r-plus",
            "command-r7b",
            "qwen2.5",
            "qwen2.5-coder",
            "qwen3",
            "devstral",
        }
        default_mode = (
            Mode.TOOLS
            if any(name in model_name.lower() for name in tool_models)
            else Mode.JSON
        )
        return from_openai(
            client, model=model_name, mode=mode or default_mode, **kwargs
        )

    return _COMPAT_BUILDERS[provider](
        provider=provider,
        model_name=model_name,
        async_client=async_client,
        mode=mode,
        api_key=api_key,
        kwargs=kwargs,
    )

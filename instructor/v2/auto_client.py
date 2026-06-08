from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any, Literal, Protocol, Union, cast, overload

from instructor.cache import BaseCache
from instructor.models import KnownModelName
from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.provider_specs import ALIAS_TO_PROVIDER, PROVIDER_SPECS

logger = logging.getLogger("instructor.auto_client")


class _BuilderModule(Protocol):
    build_from_model: Callable[..., Instructor | AsyncInstructor]


# Canonical strings and compatibility aliases accepted by from_provider().
supported_providers = list(ALIAS_TO_PROVIDER)

# Compatibility introspection for callers that previously inspected the private
# dispatch table. Runtime routing remains exclusively ProviderSpec-driven.
_PROVIDER_BUILDERS = {
    alias: PROVIDER_SPECS[provider].model_builder_module
    for alias, provider in ALIAS_TO_PROVIDER.items()
    if PROVIDER_SPECS[provider].model_builder_module is not None
}


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_provider(
    model: Union[str, KnownModelName],  # noqa: UP007
    async_client: bool = False,
    cache: BaseCache | None = None,
    mode: Union[Mode, None] = None,  # noqa: ARG001, UP007
    **kwargs: Any,
) -> Union[Instructor, AsyncInstructor]:  # noqa: UP007
    """Create an Instructor client from a model string.

    Args:
        model: String in format "provider/model-name"
              (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro")
        async_client: Whether to return an async client
        cache: Optional cache adapter (e.g., ``AutoCache`` or ``RedisCache``)
               to enable transparent response caching. Automatically flows through
               **kwargs to all provider implementations.
        mode: Override the default mode for the provider. If not specified, uses the
              recommended default mode for each provider.
        **kwargs: Additional arguments passed to the provider client functions.
                 This includes the cache parameter and any provider-specific options.

    Returns:
        Instructor or AsyncInstructor instance

    Raises:
        ValueError: If provider is not supported or model string is invalid
        ImportError: If required package for provider is not installed

    Examples:
        >>> import instructor
        >>> from instructor.cache import AutoCache
        >>>
        >>> # Basic usage
        >>> client = instructor.from_provider("openai/gpt-4")
        >>> client = instructor.from_provider("anthropic/claude-3-sonnet")
        >>>
        >>> # With caching
        >>> cache = AutoCache(maxsize=1000)
        >>> client = instructor.from_provider("openai/gpt-4", cache=cache)
        >>>
        >>> # Async clients
        >>> async_client = instructor.from_provider("openai/gpt-4", async_client=True)
    """
    # Add cache to kwargs if provided so it flows through to provider functions
    if cache is not None:
        kwargs["cache"] = cache

    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            'Model string must be in format "provider/model-name" '
            '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
        ) from None

    provider_info = {"provider": provider, "operation": "initialize"}
    logger.info(
        "Initializing %s provider with model %s",
        provider,
        model_name,
        extra=provider_info,
    )
    logger.debug(
        "Provider configuration: async_client=%s, mode=%s",
        async_client,
        mode,
        extra=provider_info,
    )
    api_key = None
    if "api_key" in kwargs:
        api_key = kwargs.pop("api_key")
        if api_key:
            logger.debug(
                "API key provided for %s provider (length: %d characters)",
                provider,
                len(api_key),
                extra=provider_info,
            )

    provider_enum = ALIAS_TO_PROVIDER.get(provider)
    spec = PROVIDER_SPECS.get(provider_enum) if provider_enum is not None else None

    if spec is None or spec.model_builder_module is None:
        from instructor.v2.core.errors import ConfigurationError

        logger.error(
            "Error initializing %s client: unsupported provider",
            provider,
            extra={**provider_info, "status": "error"},
        )
        raise ConfigurationError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )

    try:
        module = cast(
            _BuilderModule, importlib.import_module(spec.model_builder_module)
        )
        model_builder = module.build_from_model
        result = model_builder(
            provider=spec.provider,
            model_name=model_name,
            async_client=async_client,
            mode=mode,
            api_key=api_key,
            kwargs=kwargs,
        )
        logger.info("Client initialized", extra={**provider_info, "status": "success"})
        return result
    except Exception as exc:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            exc,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise

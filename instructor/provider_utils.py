"""Provider utilities for API consistency and backward compatibility.

This module provides common utilities for standardizing provider APIs while
maintaining backward compatibility with existing code.
"""

import warnings
from typing import Any, Optional, TypeVar, Union, Protocol, runtime_checkable
from functools import wraps
from enum import Enum

# Type variables
T = TypeVar("T")


class AsyncMode(Enum):
    """Standardized async mode specification."""

    SYNC = "sync"
    ASYNC = "async"


@runtime_checkable
class ProviderClient(Protocol):
    """Protocol for provider client interfaces."""

    pass


class ParameterMapping:
    """Maps legacy parameter names to standardized ones with deprecation warnings."""

    def __init__(self):
        self.mappings: dict[str, dict[str, str]] = {
            # Async parameter standardization
            "async": {
                "_async": "async_mode",
                "use_async": "async_mode",
                "is_async": "async_mode",
                "async": "async_mode",
            },
            # Model parameter standardization
            "model": {
                "model_name": "model",
                "model_id": "model",
            },
            # Retry parameter standardization
            "retry": {
                "retries": "max_retries",
                "retry_count": "max_retries",
                "max_retry": "max_retries",
            },
        }

    def normalize_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize kwargs to use standardized parameter names.

        Args:
            kwargs: Original keyword arguments

        Returns:
            Normalized kwargs with deprecation warnings issued for legacy names
        """
        normalized = kwargs.copy()

        for _category, mapping in self.mappings.items():
            for old_name, new_name in mapping.items():
                if old_name in normalized and new_name not in normalized:
                    warnings.warn(
                        f"Parameter '{old_name}' is deprecated and will be removed in v2.0. "
                        f"Use '{new_name}' instead.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    normalized[new_name] = normalized.pop(old_name)

        # Handle special async mode conversions
        if "async_mode" in normalized:
            value = normalized["async_mode"]
            if isinstance(value, bool):
                normalized["async_mode"] = AsyncMode.ASYNC if value else AsyncMode.SYNC
            elif isinstance(value, str):
                normalized["async_mode"] = AsyncMode(value)

        return normalized


# Global parameter mapper instance
parameter_mapper = ParameterMapping()


def normalize_provider_kwargs(provider: str):  # noqa: ARG001
    """Decorator to normalize provider kwargs for backward compatibility.

    Args:
        provider: Name of the provider
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Don't normalize if explicitly disabled
            if kwargs.pop("_skip_normalization", False):
                return func(*args, **kwargs)

            normalized_kwargs = parameter_mapper.normalize_kwargs(kwargs)
            return func(*args, **normalized_kwargs)

        return wrapper

    return decorator


class ProviderCapabilities:
    """Standardized provider capability declaration."""

    def __init__(
        self,
        supports_tools: bool = False,
        supports_json: bool = False,
        supports_streaming: bool = False,
        supports_parallel_tools: bool = False,
        supports_reasoning: bool = False,
        supports_structured_outputs: bool = False,
        supports_multimodal: bool = False,
        requires_max_tokens: bool = False,
        default_mode: Optional[str] = None,
    ):
        self.supports_tools = supports_tools
        self.supports_json = supports_json
        self.supports_streaming = supports_streaming
        self.supports_parallel_tools = supports_parallel_tools
        self.supports_reasoning = supports_reasoning
        self.supports_structured_outputs = supports_structured_outputs
        self.supports_multimodal = supports_multimodal
        self.requires_max_tokens = requires_max_tokens
        self.default_mode = default_mode


# Provider capability registry
PROVIDER_CAPABILITIES = {
    "openai": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_multimodal=True,
        default_mode="TOOLS",
    ),
    "anthropic": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_reasoning=True,
        supports_multimodal=True,
        requires_max_tokens=True,
        default_mode="ANTHROPIC_TOOLS",
    ),
    "gemini": ProviderCapabilities(
        supports_json=True,
        supports_streaming=True,
        supports_multimodal=True,
        default_mode="GEMINI_JSON",
    ),
    "mistral": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        supports_structured_outputs=True,
        default_mode="MISTRAL_TOOLS",
    ),
    "cohere": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        default_mode="COHERE_TOOLS",
    ),
    "groq": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        default_mode="TOOLS",
    ),
    "cerebras": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        default_mode="CEREBRAS_TOOLS",
    ),
    "fireworks": ProviderCapabilities(
        supports_json=True, supports_streaming=True, default_mode="FIREWORKS_JSON"
    ),
    "perplexity": ProviderCapabilities(
        supports_json=True, supports_streaming=True, default_mode="PERPLEXITY_JSON"
    ),
    "writer": ProviderCapabilities(
        supports_tools=True, supports_streaming=True, default_mode="WRITER_TOOLS"
    ),
    "bedrock": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        default_mode="BEDROCK_JSON",
    ),
    "vertexai": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        supports_multimodal=True,
        default_mode="VERTEXAI_TOOLS",
    ),
    "genai": ProviderCapabilities(
        supports_tools=True,
        supports_json=True,
        supports_streaming=True,
        supports_structured_outputs=True,
        supports_multimodal=True,
        default_mode="GENAI_TOOLS",
    ),
}


def get_provider_capabilities(provider: str) -> ProviderCapabilities:
    """Get capabilities for a specific provider.

    Args:
        provider: Name of the provider

    Returns:
        ProviderCapabilities instance for the provider
    """
    return PROVIDER_CAPABILITIES.get(
        provider.lower(),
        ProviderCapabilities(),  # Default with no capabilities
    )


class ClientBuilder:
    """Builder pattern for consistent client construction across providers."""

    def __init__(self, provider: str):
        self.provider = provider
        self.capabilities = get_provider_capabilities(provider)
        self._client = None
        self._mode = None
        self._async_mode = AsyncMode.SYNC
        self._kwargs = {}

    def with_client(self, client: Any) -> "ClientBuilder":
        """Set the underlying provider client."""
        self._client = client
        return self

    def with_mode(self, mode: Optional[str] = None) -> "ClientBuilder":
        """Set the mode, using provider default if not specified."""
        self._mode = mode or self.capabilities.default_mode
        return self

    def with_async_mode(
        self, async_mode: Union[bool, str, AsyncMode]
    ) -> "ClientBuilder":
        """Set async mode in a standardized way."""
        if isinstance(async_mode, bool):
            self._async_mode = AsyncMode.ASYNC if async_mode else AsyncMode.SYNC
        elif isinstance(async_mode, str):
            self._async_mode = AsyncMode(async_mode)
        else:
            self._async_mode = async_mode
        return self

    def with_kwargs(self, **kwargs) -> "ClientBuilder":
        """Set additional provider-specific kwargs."""
        self._kwargs.update(kwargs)
        return self

    def validate(self) -> None:
        """Validate the configuration is valid for this provider."""
        if self._client is None:
            raise ValueError(f"Client must be provided for {self.provider}")

        # Provider-specific validation
        if self.provider == "anthropic" and self.capabilities.requires_max_tokens:
            if "max_tokens" not in self._kwargs:
                warnings.warn(
                    "Anthropic provider requires max_tokens parameter. "
                    "Using default value of 4096.",
                    UserWarning,
                    stacklevel=2,
                )
                self._kwargs["max_tokens"] = 4096

    def build(self):
        """Build the configured client.

        This would normally return the properly configured Instructor instance,
        but we'll leave that to the actual implementation.
        """
        self.validate()
        # The actual building would be done by the specific provider module
        return {
            "provider": self.provider,
            "client": self._client,
            "mode": self._mode,
            "async_mode": self._async_mode,
            "kwargs": self._kwargs,
            "capabilities": self.capabilities,
        }


def standardize_error_message(provider: str, error_type: str, details: str = "") -> str:
    """Generate standardized error messages across providers.

    Args:
        provider: Name of the provider
        error_type: Type of error (e.g., "invalid_client", "missing_capability")
        details: Additional error details

    Returns:
        Formatted error message
    """
    base_messages = {
        "invalid_client": f"Invalid client type for {provider} provider",
        "missing_capability": f"{provider} provider does not support this capability",
        "invalid_mode": f"Invalid mode for {provider} provider",
        "missing_parameter": f"Missing required parameter for {provider} provider",
    }

    message = base_messages.get(error_type, f"Error with {provider} provider")
    if details:
        message += f": {details}"

    return message

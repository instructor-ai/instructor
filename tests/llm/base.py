"""Base classes and utilities for unified provider testing.

This module provides a unified test infrastructure that reduces duplication
across provider test suites while maintaining flexibility for provider-specific
behavior.
"""

import os
import pytest
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable
from dataclasses import dataclass
from instructor.mode import Mode


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""

    name: str
    models: list[str]
    modes: list[Mode]
    client_factory: Callable[[Any], Any]
    supports_async: bool = True
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_json: bool = True
    supports_reasoning: bool = False
    requires_max_tokens: bool = False
    skip_models: Optional[list[str]] = None
    skip_modes: Optional[list[Mode]] = None
    env_var: str = None  # e.g., "OPENAI_API_KEY"

    def should_skip_model(self, model: str) -> bool:
        """Check if a model should be skipped."""
        return model in (self.skip_models or [])

    def should_skip_mode(self, mode: Mode) -> bool:
        """Check if a mode should be skipped."""
        return mode in (self.skip_modes or [])


class BaseProviderTest(ABC):
    """Base class for provider-specific test suites.

    Subclasses should implement provider_config property and any
    provider-specific test overrides.
    """

    @property
    @abstractmethod
    def provider_config(self) -> ProviderConfig:
        """Return the provider configuration."""
        pass

    def get_client(self, fixture_client: Any, mode: Mode) -> Any:
        """Get an instructor client for the provider."""
        return self.provider_config.client_factory(fixture_client, mode)

    def skip_if_no_api_key(self):
        """Skip test if API key is not available."""
        if self.provider_config.env_var:
            if not os.environ.get(self.provider_config.env_var):
                pytest.skip(f"Skipping test, {self.provider_config.env_var} not set")

    def get_test_models(self) -> list[str]:
        """Get list of models to test."""
        # Allow limiting models via environment variable
        limit_models = os.environ.get("LIMIT_PROVIDER_MODELS")
        if limit_models:
            return [self.provider_config.models[0]]  # Just test first model
        return self.provider_config.models

    def get_test_modes(self) -> list[Mode]:
        """Get list of modes to test."""
        return self.provider_config.modes


class BaseClientFixtures:
    """Factory for creating common client fixtures."""

    @staticmethod
    def create_client_fixture(
        ClientClass: type,
        AsyncClientClass: Optional[type] = None,
        api_key_env: str = None,
        **default_kwargs,
    ):
        """Create a client fixture with common setup.

        Args:
            ClientClass: The sync client class
            AsyncClientClass: The async client class (if different)
            api_key_env: Environment variable name for API key
            **default_kwargs: Default kwargs for client initialization
        """

        @pytest.fixture(scope="module")
        def client():
            # Skip if no API key
            if api_key_env and not os.environ.get(api_key_env):
                pytest.skip(f"Skipping test, {api_key_env} not set")

            kwargs = default_kwargs.copy()

            # Check for Braintrust proxy
            if os.environ.get("BRAINTRUST_API_KEY"):
                from braintrust import init_logger

                logger = init_logger(project="instructor")
                logger._logs_queue = []

                # Most providers use base_url
                if "base_url" not in kwargs:
                    kwargs["base_url"] = logger.span.start_webdriver_server()

                # Update API key if using proxy
                if api_key_env:
                    kwargs["api_key"] = "braintrust"

            return ClientClass(**kwargs)

        @pytest.fixture(scope="module")
        async def async_client():
            # Skip if no API key
            if api_key_env and not os.environ.get(api_key_env):
                pytest.skip(f"Skipping test, {api_key_env} not set")

            kwargs = default_kwargs.copy()

            # Check for Braintrust proxy
            if os.environ.get("BRAINTRUST_API_KEY"):
                from braintrust import init_logger

                logger = init_logger(project="instructor")
                logger._logs_queue = []

                # Most providers use base_url
                if "base_url" not in kwargs:
                    kwargs["base_url"] = logger.span.start_webdriver_server()

                # Update API key if using proxy
                if api_key_env:
                    kwargs["api_key"] = "braintrust"

            ClientToUse = AsyncClientClass or ClientClass
            return ClientToUse(**kwargs)

        return client, async_client


class SharedTestModels:
    """Common test models used across providers."""

    @staticmethod
    def parametrize_models_and_modes(test_func):
        """Decorator to parametrize test with models and modes from provider config."""

        def wrapper(self, *args, **kwargs):
            provider_config = self.provider_config
            models = self.get_test_models()
            modes = self.get_test_modes()

            for model in models:
                if provider_config.should_skip_model(model):
                    continue
                for mode in modes:
                    if provider_config.should_skip_mode(mode):
                        continue
                    # Call the test function with model and mode
                    test_func(self, *args, model=model, mode=mode, **kwargs)

        return wrapper


def create_provider_fixture(provider_config: ProviderConfig):
    """Create a fixture that provides the provider configuration."""

    @pytest.fixture
    def provider():
        return provider_config

    return provider


def skip_if_missing_features(*features: str):
    """Decorator to skip tests if provider doesn't support required features.

    Args:
        *features: Feature names to check (e.g., "tools", "streaming", "reasoning")
    """

    def decorator(test_func):
        def wrapper(self, *args, **kwargs):
            provider_config = self.provider_config

            feature_map = {
                "tools": provider_config.supports_tools,
                "json": provider_config.supports_json,
                "streaming": provider_config.supports_streaming,
                "async": provider_config.supports_async,
                "reasoning": provider_config.supports_reasoning,
            }

            for feature in features:
                if not feature_map.get(feature, False):
                    pytest.skip(
                        f"Provider {provider_config.name} doesn't support {feature}"
                    )

            return test_func(self, *args, **kwargs)

        return wrapper

    return decorator

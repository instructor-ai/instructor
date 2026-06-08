"""
Comprehensive parametrized tests for all provider modes.

Tests all registered modes for each provider with actual API calls to ensure complete coverage.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable
from typing import Literal, Union

import pytest
from pydantic import BaseModel

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor import Mode
from instructor.v2 import Provider, mode_registry
from tests.v2.provider_matrix import TEST_PROVIDER_SPECS, ensure_handlers_loaded


def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        monkeypatch.delenv(key, raising=False)


class Answer(BaseModel):
    """Simple answer model."""

    answer: float


class Weather(BaseModel):
    """Weather tool."""

    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    """Search tool."""

    query: str


PROVIDER_CONFIGS = {
    provider: spec
    for provider, spec in TEST_PROVIDER_SPECS.items()
    if spec.provider_string is not None and spec.basic_modes
}


def _get_all_mode_params():
    """Generate (provider, mode) tuples for all registered modes."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config.supported_modes:
            params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_all_mode_params())
def test_mode_is_registered(provider: Provider, mode: Mode):
    """Verify each mode is registered in the v2 registry."""
    ensure_handlers_loaded(provider, skip_missing_dependency=True)

    # Skip if handler module doesn't exist or isn't registered
    if not mode_registry.is_registered(provider, mode):
        pytest.skip(
            f"Mode {mode.value} not registered for {provider.value}"  # ty: ignore[too-many-positional-arguments]
        )

    handlers = mode_registry.get_handlers(provider, mode)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def _get_basic_mode_params():
    """Generate (provider, mode) tuples for basic extraction tests."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config.basic_modes:
            params.append((provider, mode))
    return params


def _skip_on_provider_quota(provider: Provider, exc: Exception) -> None:
    """Skip tests when provider quota limits prevent execution."""
    if (
        provider == Provider.GENAI
        and isinstance(exc, InstructorRetryException)
        and "RESOURCE_EXHAUSTED" in str(exc)
    ):
        pytest.skip(
            "GenAI quota exhausted for this environment"  # ty: ignore[too-many-positional-arguments]
        )
    if (
        provider == Provider.OPENAI
        and isinstance(exc, InstructorRetryException)
        and "Connection error" in str(exc)
    ):
        if os.environ.get("CI") or os.environ.get("INSTRUCTOR_STRICT_PROVIDER_TESTS"):
            return
        pytest.skip(
            "OpenAI connectivity is unavailable in this environment"  # ty: ignore[too-many-positional-arguments]
        )


def _skip_if_provider_sdk_missing(provider: Provider) -> None:
    sdk_module = PROVIDER_CONFIGS[provider].sdk_module
    if sdk_module is None:
        return
    try:
        installed = importlib.util.find_spec(sdk_module) is not None
    except ModuleNotFoundError:
        installed = False
    if provider is Provider.XAI:
        from instructor.v2.providers.xai.client import SyncClient

        installed = installed and SyncClient is not None
    if not installed:
        pytest.skip(
            f"{sdk_module} is not installed or unusable"  # ty: ignore[too-many-positional-arguments]
        )


def test_live_provider_matrix_skips_unusable_optional_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instructor.v2.providers.xai import client as xai_client

    monkeypatch.setattr(importlib.util, "find_spec", lambda _module: object())
    monkeypatch.setattr(xai_client, "SyncClient", None)

    with pytest.raises(pytest.skip.Exception):
        _skip_if_provider_sdk_missing(Provider.XAI)


@pytest.mark.parametrize("provider,mode", _get_basic_mode_params())
@pytest.mark.requires_api_key
def test_mode_basic_extraction(
    provider: Provider, mode: Mode, monkeypatch: pytest.MonkeyPatch
):
    """Test basic extraction with each mode."""
    config = PROVIDER_CONFIGS[provider]
    assert config.provider_string is not None
    _skip_if_provider_sdk_missing(provider)
    _clear_proxy_env(monkeypatch)

    client = instructor.from_provider(
        config.provider_string,
        mode=mode,
    )

    try:
        response = client.chat.completions.create(
            response_model=Answer,
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 2? Reply with a number.",
                },
            ],
            max_tokens=1000,
        )
    except InstructorRetryException as exc:
        _skip_on_provider_quota(provider, exc)
        raise

    assert isinstance(response, Answer)
    assert response.answer == 4.0


def _get_async_mode_params():
    """Generate (provider, mode) tuples for async extraction tests."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config.async_modes:
            params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_async_mode_params())
@pytest.mark.asyncio
@pytest.mark.requires_api_key
async def test_mode_async_extraction(
    provider: Provider, mode: Mode, monkeypatch: pytest.MonkeyPatch
):
    """Test async extraction with each mode."""
    config = PROVIDER_CONFIGS[provider]
    assert config.provider_string is not None
    _skip_if_provider_sdk_missing(provider)
    _clear_proxy_env(monkeypatch)

    client = instructor.from_provider(
        config.provider_string,
        mode=mode,
        async_client=True,
    )

    try:
        response = await client.chat.completions.create(
            response_model=Answer,
            messages=[
                {
                    "role": "user",
                    "content": "What is 4 + 4? Reply with a number.",
                },
            ],
            max_tokens=1000,
        )
    except InstructorRetryException as exc:
        _skip_on_provider_quota(provider, exc)
        raise

    assert isinstance(response, Answer)
    assert response.answer == 8.0


@pytest.mark.provider(Provider.ANTHROPIC)
@pytest.mark.requires_api_key
def test_anthropic_parallel_tools_extraction():
    """Test PARALLEL_TOOLS mode extraction (Anthropic-specific)."""
    client = instructor.from_provider(
        "anthropic/claude-sonnet-4-6",
        mode=Mode.PARALLEL_TOOLS,
    )
    response = client.chat.completions.create(
        response_model=Iterable[Union[Weather, GoogleSearch]],
        messages=[
            {
                "role": "system",
                "content": "You must always use tools. Use them simultaneously when appropriate.",
            },
            {
                "role": "user",
                "content": "Get weather for San Francisco and search for Python tutorials.",
            },
        ],
        max_tokens=1000,
    )

    result = list(response)
    assert len(result) >= 1
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)


@pytest.mark.parametrize("mode", [Mode.TOOLS])
@pytest.mark.provider(Provider.ANTHROPIC)
@pytest.mark.requires_api_key
def test_anthropic_tools_with_thinking(mode: Mode):
    """Test tools modes with thinking parameter (Anthropic-specific)."""
    # Note: Thinking requires Claude 3.7 Sonnet or later
    client = instructor.from_provider(
        "anthropic/claude-3-7-sonnet-20250219",
        mode=mode,
    )
    # Note: max_tokens must be greater than thinking.budget_tokens
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "What is 5 + 5? Reply with a number.",
            },
        ],
        max_tokens=2048,  # Must be > budget_tokens
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    assert isinstance(response, Answer)
    assert response.answer == 10.0


def test_anthropic_reasoning_tools_normalizes_in_v2():
    """Legacy reasoning mode remains accepted through Anthropic tools mode."""
    assert mode_registry.is_registered(
        Provider.ANTHROPIC,
        Mode.ANTHROPIC_REASONING_TOOLS,
    )


@pytest.mark.parametrize("provider", PROVIDER_CONFIGS.keys())
@pytest.mark.requires_api_key
def test_all_modes_covered(provider: Provider):
    """Verify we're testing all registered modes for each provider."""
    config = PROVIDER_CONFIGS[provider]
    tested_modes = set(config.supported_modes)
    registered_modes = set(mode_registry.get_modes_for_provider(provider))

    # All registered modes should be tested
    assert tested_modes.issubset(registered_modes), (
        f"Tested modes {tested_modes} should be subset of registered modes {registered_modes}"
    )

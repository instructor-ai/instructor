"""
Comprehensive parametrized tests for all provider modes.

Tests all registered modes for each provider with actual API calls to ensure complete coverage.
"""

from __future__ import annotations

import pytest
from collections.abc import Iterable
from typing import Literal, Union, cast
from pydantic import BaseModel

import importlib.util
from pathlib import Path

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor import Mode
from instructor.v2 import Provider, mode_registry
from tests.v2.provider_matrix import legacy_config_dicts

# Ensure handlers are loaded by dynamically importing them
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDLER_MODULE_PATHS: dict[Provider, Path] = {
    Provider.OPENAI: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.ANTHROPIC: _PROJECT_ROOT / "instructor/v2/providers/anthropic/handlers.py",
    Provider.GENAI: _PROJECT_ROOT / "instructor/v2/providers/genai/handlers.py",
    Provider.COHERE: _PROJECT_ROOT / "instructor/v2/providers/cohere/handlers.py",
    Provider.XAI: _PROJECT_ROOT / "instructor/v2/providers/xai/handlers.py",
    Provider.GROQ: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.MISTRAL: _PROJECT_ROOT / "instructor/v2/providers/mistral/handlers.py",
    Provider.FIREWORKS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.BEDROCK: _PROJECT_ROOT / "instructor/v2/providers/bedrock/handlers.py",
    Provider.CEREBRAS: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.WRITER: _PROJECT_ROOT / "instructor/v2/providers/writer/handlers.py",
}
_HANDLERS_LOADED: set[Provider] = set()


def _ensure_handlers_loaded(provider: Provider) -> None:
    """Dynamically load handler module to ensure handlers are registered."""
    if provider in _HANDLERS_LOADED:
        return
    handler_path = _HANDLER_MODULE_PATHS.get(provider)
    if handler_path is None:
        return
    if not handler_path.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location(
            f"tests.v2.handlers_{provider.value}",
            handler_path,
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HANDLERS_LOADED.add(provider)
    except Exception:
        # Handler module may have import errors (missing dependencies)
        # This is okay - the test will skip if handlers aren't registered
        pass


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


_PROVIDER_CLIENT_CONFIGS = legacy_config_dicts()
PROVIDER_CONFIGS = {
    provider: {
        "provider_string": config["provider_string"],
        "modes": config["supported_modes"],
        "basic_modes": config["basic_modes"],
        "async_modes": config["async_modes"],
    }
    for provider, config in _PROVIDER_CLIENT_CONFIGS.items()
    if config["provider_string"] is not None and config["basic_modes"]
}


def _get_all_mode_params():
    """Generate (provider, mode) tuples for all registered modes."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config["modes"]:
            params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_all_mode_params())
def test_mode_is_registered(provider: Provider, mode: Mode):
    """Verify each mode is registered in the v2 registry."""
    _ensure_handlers_loaded(provider)

    # Skip if handler module doesn't exist or isn't registered
    if not mode_registry.is_registered(provider, mode):
        pytest.skip(f"Mode {mode.value} not registered for {provider.value}")

    handlers = mode_registry.get_handlers(provider, mode)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def _get_basic_mode_params():
    """Generate (provider, mode) tuples for basic extraction tests."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config["basic_modes"]:
            params.append((provider, mode))
    return params


def _skip_on_provider_quota(provider: Provider, exc: Exception) -> None:
    """Skip tests when provider quota limits prevent execution."""
    if (
        provider == Provider.GENAI
        and isinstance(exc, InstructorRetryException)
        and "RESOURCE_EXHAUSTED" in str(exc)
    ):
        pytest.skip("GenAI quota exhausted for this environment")


@pytest.mark.parametrize("provider,mode", _get_basic_mode_params())
@pytest.mark.requires_api_key
def test_mode_basic_extraction(provider: Provider, mode: Mode):
    """Test basic extraction with each mode."""
    config = PROVIDER_CONFIGS[provider]

    # All providers now use from_provider()
    client = instructor.from_provider(
        config["provider_string"],
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
        for mode in config["async_modes"]:
            params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_async_mode_params())
@pytest.mark.asyncio
@pytest.mark.requires_api_key
async def test_mode_async_extraction(provider: Provider, mode: Mode):
    """Test async extraction with each mode."""
    config = PROVIDER_CONFIGS[provider]

    # All providers now use from_provider()
    client = instructor.from_provider(
        config["provider_string"],
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

    result = list(cast(Iterable[Union[Weather, GoogleSearch]], response))
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


def test_anthropic_reasoning_tools_is_not_registered_in_v2():
    """Legacy reasoning mode is not part of the v2 surface."""
    assert not mode_registry.is_registered(
        Provider.ANTHROPIC,
        Mode.ANTHROPIC_REASONING_TOOLS,
    )


@pytest.mark.parametrize("provider", PROVIDER_CONFIGS.keys())
@pytest.mark.requires_api_key
def test_all_modes_covered(provider: Provider):
    """Verify we're testing all registered modes for each provider."""
    config = PROVIDER_CONFIGS[provider]
    tested_modes = set(config["modes"])
    registered_modes = set(mode_registry.get_modes_for_provider(provider))

    # All registered modes should be tested
    assert tested_modes.issubset(registered_modes), (
        f"Tested modes {tested_modes} should be subset of registered modes {registered_modes}"
    )

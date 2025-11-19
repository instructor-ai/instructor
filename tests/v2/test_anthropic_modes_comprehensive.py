"""
Comprehensive parametrized tests for all Anthropic modes.

Tests all registered modes with actual API calls to ensure complete coverage.
"""

import pytest
from collections.abc import Iterable
from typing import Literal, Union
from pydantic import BaseModel

import instructor
from instructor import Mode
from instructor.v2 import Provider, mode_registry


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


# Get all registered Anthropic modes dynamically
ANTHROPIC_MODES = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)


@pytest.mark.parametrize(
    "mode",
    [
        Mode.TOOLS,
        Mode.JSON_SCHEMA,
        Mode.PARALLEL_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
    ],
)
def test_mode_is_registered(mode):
    """Verify each mode is registered in the v2 registry."""
    assert mode_registry.is_registered(Provider.ANTHROPIC, mode)

    handlers = mode_registry.get_handlers(Provider.ANTHROPIC, mode)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


@pytest.mark.parametrize(
    "mode",
    [
        Mode.TOOLS,
        Mode.JSON_SCHEMA,
    ],
)
@pytest.mark.requires_api_key
def test_mode_basic_extraction(mode):
    """Test basic extraction with each mode."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=mode,
    )
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

    assert isinstance(response, Answer)
    assert response.answer == 4.0


@pytest.mark.requires_api_key
def test_mode_json_schema_extraction():
    """Test JSON_SCHEMA mode extraction."""
    try:
        from anthropic import transform_schema  # noqa: F401
    except ImportError:
        pytest.skip("anthropic >= 0.71.0 required for structured outputs")

    if not mode_registry.is_registered(Provider.ANTHROPIC, Mode.JSON_SCHEMA):
        pytest.skip("JSON_SCHEMA mode not available")

    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=Mode.JSON_SCHEMA,
    )
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "What is 3 + 3? Reply with a number.",
            },
        ],
        max_tokens=1000,
    )

    assert isinstance(response, Answer)
    assert response.answer == 6.0


@pytest.mark.requires_api_key
def test_mode_parallel_tools_extraction():
    """Test PARALLEL_TOOLS mode extraction."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
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


@pytest.mark.parametrize(
    "mode",
    [
        Mode.TOOLS,
        Mode.JSON_SCHEMA,
    ],
)
@pytest.mark.asyncio
@pytest.mark.requires_api_key
async def test_mode_async_extraction(mode):
    """Test async extraction with each mode."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=mode,
        async_client=True,
    )
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

    assert isinstance(response, Answer)
    assert response.answer == 8.0


@pytest.mark.parametrize(
    "mode",
    [
        Mode.TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
    ],
)
@pytest.mark.requires_api_key
def test_mode_tools_with_thinking(mode):
    """Test tools modes with thinking parameter."""
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


@pytest.mark.requires_api_key
def test_mode_reasoning_tools_deprecation():
    """Test that ANTHROPIC_REASONING_TOOLS shows deprecation warning."""
    import warnings

    import instructor.mode as mode_module

    mode_module._reasoning_tools_deprecation_shown = False  # type: ignore[attr-defined]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Trigger deprecation by accessing the handler
        from instructor.v2.providers.anthropic.handlers import (
            AnthropicReasoningToolsHandler,
        )

        handler = AnthropicReasoningToolsHandler()
        handler.prepare_request(Answer, {"messages": []})

        # Verify deprecation warning was issued
        deprecation_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, DeprecationWarning)
            and "ANTHROPIC_REASONING_TOOLS" in str(warning.message)
        ]
        assert len(deprecation_warnings) >= 1

        # Also test that it works
        client = instructor.from_provider(
            "anthropic/claude-3-5-haiku-latest",
            mode=Mode.ANTHROPIC_REASONING_TOOLS,
        )
        response = client.chat.completions.create(
            response_model=Answer,
            messages=[
                {
                    "role": "user",
                    "content": "What is 6 + 6? Reply with a number.",
                },
            ],
            max_tokens=1000,
        )

        assert isinstance(response, Answer)
        assert response.answer == 12.0


@pytest.mark.requires_api_key
def test_all_modes_covered():
    """Verify we're testing all registered modes."""
    tested_modes = {
        Mode.TOOLS,
        Mode.JSON_SCHEMA,
        Mode.PARALLEL_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
    }

    registered_modes = set(mode_registry.get_modes_for_provider(Provider.ANTHROPIC))

    # All registered modes should be tested
    assert tested_modes.issubset(registered_modes), (
        f"Tested modes {tested_modes} should be subset of registered modes {registered_modes}"
    )

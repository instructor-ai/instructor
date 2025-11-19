"""Integration tests for v2 Anthropic implementation.

Tests that verify the v2 hierarchical registry works end-to-end with Anthropic.
"""

import pytest
from pydantic import BaseModel

from instructor import Mode
from instructor.v2 import Provider, from_anthropic, mode_registry


class User(BaseModel):
    """Test model for extraction."""

    name: str
    age: int


def test_anthropic_tools_mode_registered():
    """Verify Anthropic TOOLS mode is registered in v2 registry."""
    assert mode_registry.is_registered(Provider.ANTHROPIC, Mode.TOOLS)

    handlers = mode_registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def test_anthropic_json_mode_registered():
    """Verify Anthropic JSON mode is registered in v2 registry."""
    assert mode_registry.is_registered(Provider.ANTHROPIC, Mode.ANTHROPIC_JSON)

    handlers = mode_registry.get_handlers(
        Provider.ANTHROPIC, Mode.ANTHROPIC_JSON
    )
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_v2_from_anthropic_tools_mode():
    """Test v2 from_anthropic() with TOOLS mode."""
    import anthropic

    client = anthropic.Anthropic()
    instructor_client = from_anthropic(client, Mode.TOOLS)

    assert instructor_client is not None


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_v2_from_anthropic_json_mode():
    """Test v2 from_anthropic() with JSON mode."""
    import anthropic

    client = anthropic.Anthropic()
    instructor_client = from_anthropic(client, Mode.ANTHROPIC_JSON)

    assert instructor_client is not None


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_v2_from_anthropic_async():
    """Test v2 from_anthropic() with async client."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    instructor_client = from_anthropic(client, Mode.TOOLS)

    assert instructor_client is not None
    # Should return AsyncInstructor for async client
    from instructor import AsyncInstructor

    assert isinstance(instructor_client, AsyncInstructor)


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_v2_from_anthropic_invalid_mode():
    """Test v2 from_anthropic() with unregistered mode type."""
    import anthropic

    client = anthropic.Anthropic()

    # PARALLEL_TOOLS not implemented in v2 yet
    with pytest.raises(ValueError, match="not registered"):
        from_anthropic(client, Mode.ANTHROPIC_PARALLEL_TOOLS)


def test_query_anthropic_modes():
    """Test querying v2 registry for Anthropic modes."""
    modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)

    assert Mode.TOOLS in modes
    assert Mode.ANTHROPIC_JSON in modes
    assert Mode.ANTHROPIC_REASONING_TOOLS in modes
    assert Mode.ANTHROPIC_PARALLEL_TOOLS in modes
    assert len(modes) == 4


def test_query_tools_providers():
    """Test querying v2 registry for TOOLS mode providers."""
    providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

    assert Provider.ANTHROPIC in providers


def test_anthropic_reasoning_tools_mode_registered():
    """Verify Anthropic REASONING_TOOLS mode is registered and deprecated."""
    assert mode_registry.is_registered(
        Provider.ANTHROPIC, Mode.ANTHROPIC_REASONING_TOOLS
    )

    handlers = mode_registry.get_handlers(
        Provider.ANTHROPIC, Mode.ANTHROPIC_REASONING_TOOLS
    )
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def test_anthropic_reasoning_tools_deprecation_warning():
    """Verify deprecation warning is shown for REASONING_TOOLS mode."""
    import warnings

    # The deprecation warning should be triggered when accessing the handler
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Access via handler will trigger the prepare_request which shows warning
        handlers = mode_registry.get_handlers(
            Provider.ANTHROPIC, Mode.ANTHROPIC_REASONING_TOOLS
        )
        # Create a dummy prepare_request call to trigger the deprecation warning
        from instructor.v2.providers.anthropic.handlers import (
            AnthropicReasoningToolsHandler,
        )

        handler = AnthropicReasoningToolsHandler()
        handler.prepare_request(User, {"messages": []})

        # Verify deprecation warning was issued
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ANTHROPIC_REASONING_TOOLS is deprecated" in str(w[0].message)


def test_tools_mode_with_thinking_parameter():
    """Verify TOOLS mode handles thinking parameter for extended thinking."""
    from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler

    handler = AnthropicToolsHandler()

    # Test with thinking enabled - should use auto tool_choice
    kwargs = {
        "messages": [{"role": "user", "content": "test"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    _, result = handler.prepare_request(User, kwargs)

    # Should have auto tool_choice when thinking is enabled
    assert result["tool_choice"]["type"] == "auto"
    # Should have guidance system message added
    assert "system" in result
    assert "tool call" in str(result["system"]).lower()


def test_tools_mode_without_thinking_parameter():
    """Verify TOOLS mode forces tool choice when thinking is disabled."""
    from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler

    handler = AnthropicToolsHandler()

    # Test without thinking - should use forced tool_choice
    kwargs = {
        "messages": [{"role": "user", "content": "test"}],
    }

    _, result = handler.prepare_request(User, kwargs)

    # Should have forced tool_choice when thinking is disabled
    assert result["tool_choice"]["type"] == "tool"
    assert result["tool_choice"]["name"] == "User"


def test_tools_mode_respects_user_tool_choice():
    """Verify TOOLS mode respects user-provided tool_choice parameter."""
    from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler

    handler = AnthropicToolsHandler()

    # Test with user-provided tool_choice - should not override
    kwargs = {
        "messages": [{"role": "user", "content": "test"}],
        "tool_choice": {"type": "auto"},
    }

    _, result = handler.prepare_request(User, kwargs)

    # Should respect user's tool_choice
    assert result["tool_choice"]["type"] == "auto"

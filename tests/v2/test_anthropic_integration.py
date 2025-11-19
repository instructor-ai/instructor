"""Integration tests for v2 Anthropic implementation.

Tests that verify the v2 hierarchical registry works end-to-end with Anthropic.
"""

import pytest
from pydantic import BaseModel

from instructor import Mode
from instructor.v2 import Provider, from_anthropic, mode_registry

# All tests in this module require API key
pytestmark = pytest.mark.requires_api_key


class User(BaseModel):
    """Test model for extraction."""

    name: str
    age: int


def test_v2_from_anthropic_tools_mode():
    """Test v2 from_anthropic() with TOOLS mode."""
    import anthropic

    client = anthropic.Anthropic()
    instructor_client = from_anthropic(client, Mode.TOOLS)

    assert instructor_client is not None


def test_v2_from_anthropic_json_mode():
    """Test v2 from_anthropic() with JSON mode."""
    import anthropic

    client = anthropic.Anthropic()
    # Use generic Mode.JSON instead of Mode.ANTHROPIC_JSON
    instructor_client = from_anthropic(client, Mode.JSON)

    assert instructor_client is not None


def test_v2_from_anthropic_async():
    """Test v2 from_anthropic() with async client."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    instructor_client = from_anthropic(client, Mode.TOOLS)

    assert instructor_client is not None
    # Should return AsyncInstructor for async client
    from instructor import AsyncInstructor

    assert isinstance(instructor_client, AsyncInstructor)


def test_v2_from_anthropic_invalid_mode():
    """Test v2 from_anthropic() with unregistered mode type."""
    import anthropic
    from instructor.core.exceptions import ModeError

    client = anthropic.Anthropic()

    # Use a mode that doesn't exist or isn't registered for Anthropic
    # Note: ANTHROPIC_PARALLEL_TOOLS normalizes to PARALLEL_TOOLS which IS registered
    # So we need to use a truly invalid mode
    with pytest.raises(ModeError, match="Invalid mode"):
        from_anthropic(client, Mode.MD_JSON)  # MD_JSON is not registered for Anthropic


def test_query_anthropic_modes():
    """Test querying v2 registry for Anthropic modes."""
    modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)

    assert Mode.TOOLS in modes
    assert Mode.JSON in modes  # Generic mode, not ANTHROPIC_JSON
    assert Mode.ANTHROPIC_REASONING_TOOLS in modes
    assert Mode.PARALLEL_TOOLS in modes  # Generic mode, not ANTHROPIC_PARALLEL_TOOLS
    # May also have JSON_SCHEMA if structured outputs is available
    assert len(modes) >= 4


def test_query_tools_providers():
    """Test querying v2 registry for TOOLS mode providers."""
    providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

    assert Provider.ANTHROPIC in providers


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

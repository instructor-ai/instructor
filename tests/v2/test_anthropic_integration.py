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
    assert mode_registry.is_registered(Provider.ANTHROPIC, Mode.JSON)

    handlers = mode_registry.get_handlers(Provider.ANTHROPIC, Mode.JSON)
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
    instructor_client = from_anthropic(client, Mode.JSON)

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
    assert Mode.JSON in modes
    assert len(modes) == 2  # Only TOOLS and JSON for now


def test_query_tools_providers():
    """Test querying v2 registry for TOOLS mode providers."""
    providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

    assert Provider.ANTHROPIC in providers

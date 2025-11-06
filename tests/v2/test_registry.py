"""Tests for v2 mode registry."""

import pytest

from instructor.v2.core import ModeType, Provider, mode_registry
from instructor.v2.core.decorators import register_mode_handler


def test_registry_registration():
    """Test basic registration."""

    @register_mode_handler(Provider.OPENAI, ModeType.TOOLS)
    class TestHandler:
        def prepare_request(self, response_model, kwargs):
            return response_model, kwargs

        def handle_reask(self, kwargs, _response, _exception):
            return kwargs

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Check it's registered
    assert mode_registry.is_registered(Provider.OPENAI, ModeType.TOOLS)

    # Get handlers
    handlers = mode_registry.get_handlers(Provider.OPENAI, ModeType.TOOLS)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def test_registry_get_handler():
    """Test getting specific handler types."""

    @register_mode_handler(Provider.GEMINI, ModeType.JSON)
    class TestHandler:
        def prepare_request(self, response_model, _kwargs):
            return response_model, {"test": "request"}

        def handle_reask(self, _kwargs, _response, _exception):
            return {"test": "reask"}

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Get individual handlers
    request_handler = mode_registry.get_handler(
        Provider.GEMINI, ModeType.JSON, "request"
    )
    result = request_handler(None, {})
    assert result[1]["test"] == "request"

    reask_handler = mode_registry.get_handler(Provider.GEMINI, ModeType.JSON, "reask")
    result = reask_handler({}, None, None)
    assert result["test"] == "reask"


def test_registry_query_by_provider():
    """Test querying modes for a provider."""
    # Anthropic should have TOOLS and JSON registered
    modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)
    assert ModeType.TOOLS in modes
    assert ModeType.JSON in modes


def test_registry_query_by_mode_type():
    """Test querying providers for a mode type."""
    # TOOLS should be supported by Anthropic (and possibly others)
    providers = mode_registry.get_providers_for_mode(ModeType.TOOLS)
    assert Provider.ANTHROPIC in providers


def test_registry_list_modes():
    """Test listing all registered modes."""
    all_modes = mode_registry.list_modes()

    # Should include Anthropic modes
    assert (Provider.ANTHROPIC, ModeType.TOOLS) in all_modes
    assert (Provider.ANTHROPIC, ModeType.JSON) in all_modes


def test_registry_not_registered():
    """Test error when mode not registered."""
    with pytest.raises(KeyError, match="not registered"):
        mode_registry.get_handlers(Provider.COHERE, ModeType.TOOLS)


def test_registry_invalid_handler_type():
    """Test error for invalid handler type."""
    with pytest.raises(ValueError, match="Invalid handler_type"):
        mode_registry.get_handler(Provider.ANTHROPIC, ModeType.TOOLS, "invalid_type")

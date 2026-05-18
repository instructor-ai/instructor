"""Tests for v2 mode registry."""

import pytest

from instructor import Mode
from instructor.v2 import Provider, mode_registry
from instructor.v2.core.decorators import register_mode_handler
from tests.v2.conftest import get_registered_provider_mode_pairs


def _get_registered_providers() -> list[Provider]:
    pairs = get_registered_provider_mode_pairs()
    return sorted({provider for provider, _ in pairs}, key=lambda p: p.value)


def _get_registered_modes() -> list[Mode]:
    pairs = get_registered_provider_mode_pairs()
    return sorted({mode for _, mode in pairs}, key=lambda m: m.value)


def _get_registered_provider_modes() -> list[tuple[Provider, Mode]]:
    return get_registered_provider_mode_pairs()


def test_registry_registration():
    """Test basic registration."""

    @register_mode_handler(Provider.DEEPSEEK, Mode.JSON)
    class TestHandler:
        def prepare_request(self, response_model, kwargs):
            return response_model, kwargs

        def handle_reask(self, kwargs, _response, _exception):
            return kwargs

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Check it's registered
    assert mode_registry.is_registered(Provider.DEEPSEEK, Mode.JSON)

    # Get handlers
    handlers = mode_registry.get_handlers(Provider.DEEPSEEK, Mode.JSON)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


def test_registry_get_handler():
    """Test getting specific handler types."""

    @register_mode_handler(Provider.OPENROUTER, Mode.TOOLS)
    class TestHandler:
        def prepare_request(self, response_model, _kwargs):
            return response_model, {"test": "request"}

        def handle_reask(self, _kwargs, _response, _exception):
            return {"test": "reask"}

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Get individual handlers
    request_handler = mode_registry.get_handler(
        Provider.OPENROUTER, Mode.TOOLS, "request"
    )
    result = request_handler(None, {})
    assert result[1]["test"] == "request"

    reask_handler = mode_registry.get_handler(Provider.OPENROUTER, Mode.TOOLS, "reask")
    result = reask_handler({}, None, None)
    assert result["test"] == "reask"


@pytest.mark.parametrize("provider", _get_registered_providers())
def test_registry_query_by_provider(provider: Provider):
    """Test querying modes for a provider."""
    modes = mode_registry.get_modes_for_provider(provider)
    assert modes, f"{provider.value} should have at least one mode"

    expected_modes = {
        mode for prov, mode in get_registered_provider_mode_pairs() if prov == provider
    }
    assert expected_modes.issubset(set(modes))


@pytest.mark.parametrize("mode", _get_registered_modes())
def test_registry_query_by_mode_type(mode: Mode):
    """Test querying providers for a mode type."""
    providers = mode_registry.get_providers_for_mode(mode)
    assert providers, f"{mode.value} should have at least one provider"

    expected_providers = {
        provider
        for provider, registered_mode in get_registered_provider_mode_pairs()
        if registered_mode == mode
    }
    assert expected_providers.issubset(set(providers))


@pytest.mark.parametrize("provider,mode", _get_registered_provider_modes())
def test_registry_list_modes(provider: Provider, mode: Mode):
    """Test listing all registered modes."""
    all_modes = mode_registry.list_modes()
    assert (provider, mode) in all_modes


def test_registry_not_registered():
    """Test error when mode not registered."""
    with pytest.raises(KeyError, match="not registered"):
        mode_registry.get_handlers(Provider.GEMINI, Mode.JSON_SCHEMA)


@pytest.mark.parametrize("provider,mode", _get_registered_provider_modes())
def test_registry_invalid_handler_type(provider: Provider, mode: Mode):
    """Test error for invalid handler type."""
    with pytest.raises(ValueError, match="Invalid handler_type"):
        mode_registry.get_handler(provider, mode, "invalid_type")

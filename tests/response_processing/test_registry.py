"""Tests for the handler registry."""

import pytest
from typing import Any

from instructor.mode import Mode
from instructor.response_processing.registry import HandlerRegistry


def dummy_handler(
    response_model: type, kwargs: dict[str, Any]
) -> tuple[type, dict[str, Any]]:
    """Dummy handler for testing."""
    kwargs["handled"] = True
    return response_model, kwargs


class TestHandlerRegistry:
    """Test the HandlerRegistry class."""

    def test_register_and_get_handler(self):
        """Test registering and retrieving a handler."""
        registry = HandlerRegistry()
        registry.register(Mode.TOOLS, dummy_handler)

        handler = registry.get_handler(Mode.TOOLS)
        assert handler is dummy_handler

    def test_get_nonexistent_handler(self):
        """Test getting a handler that doesn't exist."""
        registry = HandlerRegistry()
        handler = registry.get_handler(Mode.TOOLS)
        assert handler is None

    def test_is_registered(self):
        """Test checking if a handler is registered."""
        registry = HandlerRegistry()

        assert not registry.is_registered(Mode.TOOLS)

        registry.register(Mode.TOOLS, dummy_handler)
        assert registry.is_registered(Mode.TOOLS)

    def test_handle_with_registered_handler(self):
        """Test handling with a registered handler."""
        registry = HandlerRegistry()
        registry.register(Mode.TOOLS, dummy_handler)

        response_model = str
        kwargs = {"test": "value"}

        result_model, result_kwargs = registry.handle(
            Mode.TOOLS, response_model, kwargs
        )

        assert result_model is response_model
        assert result_kwargs["test"] == "value"
        assert result_kwargs["handled"] is True

    def test_handle_with_unregistered_mode(self):
        """Test handling with an unregistered mode."""
        registry = HandlerRegistry()

        with pytest.raises(ValueError, match="No handler registered for mode"):
            registry.handle(Mode.TOOLS, str, {})

    def test_register_multiple(self):
        """Test registering multiple handlers at once."""
        registry = HandlerRegistry()

        def handler1(rm, kw):
            return rm, kw

        def handler2(rm, kw):
            return rm, kw

        handlers = {
            Mode.TOOLS: handler1,
            Mode.JSON: handler2,
        }

        registry.register_multiple(handlers)

        assert registry.get_handler(Mode.TOOLS) is handler1
        assert registry.get_handler(Mode.JSON) is handler2

    def test_override_handler(self):
        """Test that registering a handler for the same mode overrides the previous one."""
        registry = HandlerRegistry()

        def handler1(rm, kw):
            kw["handler"] = "handler1"
            return rm, kw

        def handler2(rm, kw):
            kw["handler"] = "handler2"
            return rm, kw

        registry.register(Mode.TOOLS, handler1)
        registry.register(Mode.TOOLS, handler2)

        _, kwargs = registry.handle(Mode.TOOLS, str, {})
        assert kwargs["handler"] == "handler2"

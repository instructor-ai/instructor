"""Fireworks-specific response handlers."""

from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler, ToolsHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class FireworksToolsHandler(ToolsHandler):
    """Handler for Fireworks tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Fireworks tools mode."""
        if "stream" not in kwargs:
            kwargs["stream"] = False
        kwargs["tools"] = [self.create_tool_definition(response_model)]
        kwargs = self.set_tool_choice(kwargs, response_model.openai_schema["name"])
        return response_model, kwargs


class FireworksJSONHandler(BaseHandler):
    """Handler for Fireworks JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Fireworks JSON mode."""
        if "stream" not in kwargs:
            kwargs["stream"] = False

        kwargs["response_format"] = {
            "type": "json_object",
            "schema": response_model.model_json_schema(),
        }
        return response_model, kwargs


def register_fireworks_handlers() -> None:
    """Register all Fireworks handlers."""
    handler_registry.register(Mode.FIREWORKS_TOOLS, FireworksToolsHandler())
    handler_registry.register(Mode.FIREWORKS_JSON, FireworksJSONHandler())

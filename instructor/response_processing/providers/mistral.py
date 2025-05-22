"""Mistral-specific response handlers."""

from typing import Any, TypeVar

try:
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:
    response_format_from_pydantic_model = None

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler, ToolsHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class MistralToolsHandler(ToolsHandler):
    """Handler for Mistral tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Mistral tools mode."""
        kwargs["tools"] = [self.create_tool_definition(response_model)]
        kwargs["tool_choice"] = "any"
        return response_model, kwargs


class MistralStructuredOutputsHandler(BaseHandler):
    """Handler for Mistral structured outputs mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Mistral structured outputs mode."""
        if response_format_from_pydantic_model is None:
            raise ImportError("mistralai is not installed")
        kwargs["response_format"] = response_format_from_pydantic_model(response_model)
        kwargs.pop("tools", None)
        kwargs.pop("response_model", None)
        return response_model, kwargs


def register_mistral_handlers() -> None:
    """Register all Mistral handlers."""
    handler_registry.register(Mode.MISTRAL_TOOLS, MistralToolsHandler())
    handler_registry.register(
        Mode.MISTRAL_STRUCTURED_OUTPUTS, MistralStructuredOutputsHandler()
    )

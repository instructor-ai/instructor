"""Writer-specific response handlers."""

from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import ToolsHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class WriterToolsHandler(ToolsHandler):
    """Handler for Writer tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Writer tools mode."""
        kwargs["tools"] = [self.create_tool_definition(response_model)]
        kwargs["tool_choice"] = "auto"
        return response_model, kwargs


def register_writer_handlers() -> None:
    """Register all Writer handlers."""
    handler_registry.register(Mode.WRITER_TOOLS, WriterToolsHandler())

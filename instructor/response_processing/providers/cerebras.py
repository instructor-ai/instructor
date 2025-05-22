"""Cerebras-specific response handlers."""

from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler, ToolsHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class CerebrasToolsHandler(ToolsHandler):
    """Handler for Cerebras tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Cerebras tools mode."""
        if kwargs.get("stream", False):
            raise ValueError("Stream is not supported for Cerebras Tool Calling")
        kwargs["tools"] = [self.create_tool_definition(response_model)]
        kwargs = self.set_tool_choice(kwargs, response_model.openai_schema["name"])
        return response_model, kwargs


class CerebrasJSONHandler(BaseHandler):
    """Handler for Cerebras JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Cerebras JSON mode."""
        instruction = f"""
You are a helpful assistant that excels at following instructions.Your task is to understand the content and provide the parsed objects in json that match the following json_schema:\n

Here is the relevant JSON schema to adhere to

<schema>
{response_model.model_json_schema()}
</schema>

Your response should consist only of a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""

        kwargs["messages"] = [{"role": "system", "content": instruction}] + kwargs[
            "messages"
        ]
        return response_model, kwargs


def register_cerebras_handlers() -> None:
    """Register all Cerebras handlers."""
    handler_registry.register(Mode.CEREBRAS_TOOLS, CerebrasToolsHandler())
    handler_registry.register(Mode.CEREBRAS_JSON, CerebrasJSONHandler())

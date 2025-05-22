"""Google GenAI-specific response handlers."""

from typing import Any, TypeVar

try:
    from google.genai import types
except ImportError:
    types = None

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.registry import handler_registry
from instructor.utils import (
    convert_to_genai_messages,
    extract_genai_system_message,
    map_to_gemini_function_schema,
)

T = TypeVar("T")


class GenAIStructuredOutputsHandler(BaseHandler):
    """Handler for GenAI structured outputs mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle GenAI structured outputs mode."""
        if types is None:
            raise ImportError("google.genai is not installed")

        if kwargs.get("system"):
            system_message = kwargs.pop("system")
        elif kwargs.get("messages"):
            system_message = extract_genai_system_message(kwargs["messages"])
        else:
            system_message = None

        kwargs["contents"] = convert_to_genai_messages(kwargs["messages"])

        # Validate that the schema doesn't contain any optional fields
        map_to_gemini_function_schema(response_model.model_json_schema())

        kwargs["config"] = types.GenerateContentConfig(
            system_instruction=system_message,
            response_mime_type="application/json",
            response_schema=response_model,
        )
        kwargs.pop("response_model", None)
        kwargs.pop("messages", None)

        return response_model, kwargs


class GenAIToolsHandler(BaseHandler):
    """Handler for GenAI tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle GenAI tools mode."""
        if types is None:
            raise ImportError("google.genai is not installed")

        schema = map_to_gemini_function_schema(response_model.model_json_schema())
        function_definition = types.FunctionDeclaration(
            name=response_model.__name__,
            description=response_model.__doc__,
            parameters=schema,
        )

        if kwargs.get("system"):
            system_message = kwargs.pop("system")
        elif kwargs.get("messages"):
            system_message = extract_genai_system_message(kwargs["messages"])
        else:
            system_message = None

        kwargs["config"] = types.GenerateContentConfig(
            system_instruction=system_message,
            tools=[types.Tool(function_declarations=[function_definition])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[response_model.__name__]
                ),
            ),
        )

        kwargs["contents"] = convert_to_genai_messages(kwargs["messages"])

        kwargs.pop("response_model", None)
        kwargs.pop("messages", None)

        return response_model, kwargs


def register_genai_handlers() -> None:
    """Register all GenAI handlers."""
    handler_registry.register(
        Mode.GENAI_STRUCTURED_OUTPUTS, GenAIStructuredOutputsHandler()
    )
    handler_registry.register(Mode.GENAI_TOOLS, GenAIToolsHandler())

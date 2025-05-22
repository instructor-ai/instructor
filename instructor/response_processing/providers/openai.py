"""OpenAI-specific response handlers."""

import json
from textwrap import dedent
from typing import Any, TypeVar

from openai import pydantic_function_tool

from instructor.dsl.parallel import ParallelModel, handle_parallel_model
from instructor.exceptions import ConfigurationError
from instructor.mode import Mode
from instructor.response_processing.base import JSONHandler, ToolsHandler
from instructor.utils import merge_consecutive_messages
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


class OpenAIToolsHandler(ToolsHandler):
    """Handler for OpenAI tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OpenAI tools mode."""
        tool_def = self.create_tool_definition(response_model)
        kwargs["tools"] = [tool_def]
        kwargs = self.set_tool_choice(kwargs, tool_def["function"]["name"])
        return response_model, kwargs


class OpenAIToolsStrictHandler(ToolsHandler):
    """Handler for OpenAI strict tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OpenAI strict tools mode."""
        response_model_schema = pydantic_function_tool(response_model)
        response_model_schema["function"]["strict"] = True
        kwargs["tools"] = [response_model_schema]
        kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": response_model_schema["function"]["name"]},
        }
        return response_model, kwargs


class OpenAIFunctionsHandler(ToolsHandler):
    """Handler for OpenAI functions mode (deprecated)."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OpenAI functions mode."""
        Mode.warn_mode_functions_deprecation()
        tool_def = self.create_tool_definition(response_model)
        schema = tool_def["function"]
        kwargs["functions"] = [schema]
        kwargs["function_call"] = {"name": schema["name"]}
        return response_model, kwargs


class OpenAIJSONHandler(JSONHandler):
    """Handler for OpenAI JSON modes."""

    def __init__(self, mode: Mode):
        self.mode = mode

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OpenAI JSON mode."""
        instruction = self.create_json_instruction(response_model)

        if self.mode == Mode.JSON:
            kwargs["response_format"] = {"type": "json_object"}
        elif self.mode == Mode.JSON_SCHEMA:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": response_model.model_json_schema(),
            }
        elif self.mode == Mode.MD_JSON:
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
                }
            )
            kwargs["messages"] = merge_consecutive_messages(kwargs["messages"])

        kwargs = self.add_json_instruction_to_messages(kwargs, instruction)
        return response_model, kwargs


class OpenAIParallelToolsHandler:
    """Handler for OpenAI parallel tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OpenAI parallel tools mode."""
        if kwargs.get("stream", False):
            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )
        kwargs["tools"] = handle_parallel_model(response_model)
        kwargs["tool_choice"] = "auto"
        return ParallelModel(typehint=response_model), kwargs


class O1JSONHandler(JSONHandler):
    """Handler for O1 model JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle O1 JSON mode."""
        roles = [message["role"] for message in kwargs.get("messages", [])]
        if "system" in roles:
            raise ValueError("System messages are not supported For the O1 models")

        message = dedent(
            f"""
            Understand the content and provide
            the parsed objects in json that match the following json_schema:\n\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        kwargs["messages"].append(
            {
                "role": "user",
                "content": message,
            }
        )
        return response_model, kwargs


class ResponsesToolsHandler(ToolsHandler):
    """Handler for responses tools mode."""

    def __init__(self, with_inbuilt_tools: bool = False):
        self.with_inbuilt_tools = with_inbuilt_tools

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle responses tools mode."""
        schema = pydantic_function_tool(response_model)
        del schema["function"]["strict"]

        tool_definition = {
            "type": "function",
            "name": schema["function"]["name"],
            "parameters": schema["function"]["parameters"],
        }

        if "description" in schema["function"]:
            tool_definition["description"] = schema["function"]["description"]
        else:
            tool_definition["description"] = (
                f"Correctly extracted `{response_model.__name__}` with all "
                f"the required parameters with correct types"
            )

        if self.with_inbuilt_tools:
            if not kwargs.get("tools"):
                kwargs["tools"] = [tool_definition]
                kwargs["tool_choice"] = {
                    "type": "function",
                    "name": tool_definition["name"],
                }
            else:
                kwargs["tools"].append(tool_definition)
        else:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": schema["function"]["name"],
                    "parameters": schema["function"]["parameters"],
                }
            ]
            kwargs["tool_choice"] = {
                "type": "function",
                "name": schema["function"]["name"],
            }

        if kwargs.get("max_tokens") is not None:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        return response_model, kwargs


class OpenRouterStructuredOutputsHandler:
    """Handler for OpenRouter structured outputs."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle OpenRouter structured outputs."""
        schema = response_model.model_json_schema()
        schema["additionalProperties"] = False
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        return response_model, kwargs


def register_openai_handlers() -> None:
    """Register all OpenAI handlers."""
    handler_registry.register(Mode.TOOLS, OpenAIToolsHandler())
    handler_registry.register(Mode.TOOLS_STRICT, OpenAIToolsStrictHandler())
    handler_registry.register(Mode.FUNCTIONS, OpenAIFunctionsHandler())
    handler_registry.register(Mode.JSON, OpenAIJSONHandler(Mode.JSON))
    handler_registry.register(Mode.JSON_SCHEMA, OpenAIJSONHandler(Mode.JSON_SCHEMA))
    handler_registry.register(Mode.MD_JSON, OpenAIJSONHandler(Mode.MD_JSON))
    handler_registry.register(Mode.JSON_O1, O1JSONHandler())
    handler_registry.register(Mode.PARALLEL_TOOLS, OpenAIParallelToolsHandler())
    handler_registry.register(Mode.RESPONSES_TOOLS, ResponsesToolsHandler())
    handler_registry.register(
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        ResponsesToolsHandler(with_inbuilt_tools=True),
    )
    handler_registry.register(
        Mode.OPENROUTER_STRUCTURED_OUTPUTS, OpenRouterStructuredOutputsHandler()
    )

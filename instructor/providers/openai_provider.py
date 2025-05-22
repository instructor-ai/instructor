"""OpenAI provider implementation."""

from typing import Any, TypeVar, Union
from textwrap import dedent
import json

from pydantic import BaseModel

from ..mode import Mode
from ..client import Instructor, AsyncInstructor
from ..patch import patch, apatch
from ..utils import merge_consecutive_messages
from ..dsl.parallel import handle_parallel_model
from ..dsl.simple_type import ModelAdapter, is_simple_type
from .base import BaseProvider
from .registry import ProviderRegistry

T_Model = TypeVar("T_Model", bound=BaseModel)


@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI and compatible APIs."""

    @property
    def name(self) -> str:
        return "openai"

    def get_supported_modes(self) -> set[Mode]:
        return {
            Mode.TOOLS,
            Mode.TOOLS_STRICT,
            Mode.FUNCTIONS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.JSON_O1,
            Mode.PARALLEL_TOOLS,
        }

    def validate_response(self, response: Any, mode: Mode) -> None:
        """Validate OpenAI response format."""
        # OpenAI responses are generally well-formed
        pass

    def process_response(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process OpenAI response based on mode."""
        # This will be implemented by delegating to existing logic
        # For now, we'll import and use the existing process_response function
        from ..process_response import process_response as legacy_process

        return legacy_process(
            response, response_model=response_model, mode=mode, **kwargs
        )

    async def process_response_async(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process OpenAI response asynchronously."""
        from ..process_response import process_response_async as legacy_process_async

        return await legacy_process_async(
            response, response_model=response_model, mode=mode, **kwargs
        )

    def create_instructor(
        self, client: Any, mode: Mode, **kwargs: Any
    ) -> Union[Instructor, AsyncInstructor]:
        """Create an instructor instance for OpenAI."""
        # Use existing patch mechanism
        create_fn = kwargs.pop("create", None)
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            # Async client
            if hasattr(client.chat.completions, "acreate"):
                return apatch(create=create_fn)(client, mode=mode)
        # Sync client
        return patch(create=create_fn)(client, mode=mode)

    def prepare_request(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any], mode: Mode
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Prepare request parameters based on mode."""
        if mode == Mode.FUNCTIONS:
            return self._handle_functions(response_model, new_kwargs)
        elif mode == Mode.TOOLS_STRICT:
            return self._handle_tools_strict(response_model, new_kwargs)
        elif mode == Mode.TOOLS:
            return self._handle_tools(response_model, new_kwargs)
        elif mode == Mode.JSON_O1:
            return self._handle_json_o1(response_model, new_kwargs)
        elif mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
            return self._handle_json_modes(response_model, new_kwargs, mode)
        elif mode == Mode.PARALLEL_TOOLS:
            return self._handle_parallel_tools(response_model, new_kwargs)
        else:
            raise ValueError(f"Unsupported mode for OpenAI provider: {mode}")

    def _handle_functions(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle FUNCTIONS mode."""
        new_kwargs["functions"] = [response_model.openai_schema()]
        new_kwargs["function_call"] = {"name": response_model.openai_schema()["name"]}
        if "stream" in new_kwargs:
            del new_kwargs["stream"]
        return response_model, new_kwargs

    def _handle_tools_strict(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle TOOLS_STRICT mode."""
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]  # type: ignore

        # Generate OpenAI schema for the model
        from ..function_calls import openai_schema as create_openai_schema

        openai_schema = create_openai_schema(response_model).openai_schema
        tool_schema = {"type": "function", "function": openai_schema, "strict": True}
        new_kwargs["tools"] = [tool_schema]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": openai_schema["name"]},
        }
        return response_model, new_kwargs

    def _handle_tools(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle TOOLS mode."""
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]  # type: ignore

        # Generate OpenAI schema for the model
        from ..function_calls import openai_schema as create_openai_schema

        openai_schema = create_openai_schema(response_model).openai_schema
        tool_schema = {"type": "function", "function": openai_schema}
        new_kwargs["tools"] = [tool_schema]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": openai_schema["name"]},
        }
        return response_model, new_kwargs

    def _handle_json_o1(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle JSON_O1 mode."""
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        if "messages" not in new_kwargs:
            new_kwargs["messages"] = []

        messages = new_kwargs["messages"]
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += (
                f"\n\nReturn only a valid JSON object that matches "
                f"this schema: {schema}. Do not include any other text."
            )
        return response_model, new_kwargs

    def _handle_json_modes(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any], mode: Mode
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle JSON, JSON_SCHEMA, and MD_JSON modes."""
        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        if mode == Mode.JSON:
            new_kwargs["response_format"] = {"type": "json_object"}
        elif mode == Mode.JSON_SCHEMA:
            new_kwargs["response_format"] = {
                "type": "json_object",
                "schema": response_model.model_json_schema(),
            }
        elif mode == Mode.MD_JSON:
            new_kwargs["messages"].append(
                {
                    "role": "user",
                    "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
                }
            )
            new_kwargs["messages"] = merge_consecutive_messages(new_kwargs["messages"])

        # Add system message
        if new_kwargs["messages"][0]["role"] != "system":
            new_kwargs["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": message,
                },
            )
        elif isinstance(new_kwargs["messages"][0]["content"], str):
            new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        elif isinstance(new_kwargs["messages"][0]["content"], list):
            new_kwargs["messages"][0]["content"][0]["text"] += f"\n\n{message}"
        else:
            raise ValueError(
                "Invalid message format, must be a string or a list of messages"
            )

        return response_model, new_kwargs

    def _handle_parallel_tools(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle PARALLEL_TOOLS mode."""
        new_response_model = handle_parallel_model(response_model)
        new_kwargs["tools"] = new_response_model.openai_schema()["tools"]
        new_kwargs["tool_choice"] = "auto"
        return new_response_model, new_kwargs

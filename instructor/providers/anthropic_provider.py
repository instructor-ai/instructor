"""Anthropic provider implementation."""

from typing import Any, TypeVar, Union
from textwrap import dedent

from pydantic import BaseModel

from ..mode import Mode
from ..client import Instructor, AsyncInstructor
from ..patch import patch, apatch
from ..utils import extract_system_messages, combine_system_messages
from ..dsl.simple_type import ModelAdapter, is_simple_type
from .base import BaseProvider
from .registry import ProviderRegistry

T_Model = TypeVar("T_Model", bound=BaseModel)


@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic Claude."""

    @property
    def name(self) -> str:
        return "anthropic"

    def get_supported_modes(self) -> set[Mode]:
        return {
            Mode.ANTHROPIC_TOOLS,
            Mode.ANTHROPIC_JSON,
            Mode.ANTHROPIC_REASONING_TOOLS,
        }

    def validate_response(self, response: Any, mode: Mode) -> None:
        """Validate Anthropic response format."""
        # Anthropic-specific validation can be added here
        pass

    def process_response(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process Anthropic response based on mode."""
        # Delegate to existing logic for now
        from ..process_response import process_response as legacy_process

        return legacy_process(
            response, response_model=response_model, mode=mode, **kwargs
        )

    async def process_response_async(
        self, response: Any, response_model: type[T_Model], mode: Mode, **kwargs: Any
    ) -> T_Model:
        """Process Anthropic response asynchronously."""
        from ..process_response import process_response_async as legacy_process_async

        return await legacy_process_async(
            response, response_model=response_model, mode=mode, **kwargs
        )

    def create_instructor(
        self, client: Any, mode: Mode, **kwargs: Any
    ) -> Union[Instructor, AsyncInstructor]:
        """Create an instructor instance for Anthropic."""
        create_fn = kwargs.pop("create", None)
        # Check if it's an async client
        if hasattr(client, "messages") and hasattr(client.messages, "acreate"):
            return apatch(create=create_fn)(client, mode=mode)
        return patch(create=create_fn)(client, mode=mode)

    def prepare_request(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any], mode: Mode
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Prepare request parameters based on mode."""
        if mode == Mode.ANTHROPIC_TOOLS:
            return self._handle_anthropic_tools(response_model, new_kwargs)
        elif mode == Mode.ANTHROPIC_REASONING_TOOLS:
            return self._handle_anthropic_reasoning_tools(response_model, new_kwargs)
        elif mode == Mode.ANTHROPIC_JSON:
            return self._handle_anthropic_json(response_model, new_kwargs)
        else:
            raise ValueError(f"Unsupported mode for Anthropic provider: {mode}")

    def _handle_anthropic_tools(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle ANTHROPIC_TOOLS mode."""
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]  # type: ignore

        tool_descriptions = response_model.anthropic_schema
        new_kwargs["tools"] = [tool_descriptions]
        new_kwargs["tool_choice"] = {
            "type": "tool",
            "name": response_model.__name__,
        }

        # Extract and combine system messages
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )

        # Remove system messages from messages list
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]

        return response_model, new_kwargs

    def _handle_anthropic_reasoning_tools(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle ANTHROPIC_REASONING_TOOLS mode."""
        # First apply standard tools handling
        response_model, new_kwargs = self._handle_anthropic_tools(
            response_model, new_kwargs
        )

        # Reasoning does not allow forced tool use
        new_kwargs["tool_choice"] = {"type": "auto"}

        # Add message recommending tool use
        implicit_forced_tool_message = dedent(
            """
            Return only the tool call and no additional text.
            """
        )
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"),
            [{"type": "text", "text": implicit_forced_tool_message}],
        )

        return response_model, new_kwargs

    def _handle_anthropic_json(
        self, response_model: type[T_Model], new_kwargs: dict[str, Any]
    ) -> tuple[type[T_Model], dict[str, Any]]:
        """Handle ANTHROPIC_JSON mode."""
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]  # type: ignore

        # Extract and format schema
        schema = response_model.model_json_schema()

        # Create system message
        json_system_messages = dedent(
            f"""
            You are a helpful assistant that responds with valid JSON.
            Your response must match this JSON schema:
            {schema}
            """
        )

        # Extract existing system messages
        system_messages = extract_system_messages(new_kwargs.get("messages", []))

        # Combine all system messages
        all_system_content = [{"type": "text", "text": json_system_messages}]
        if system_messages:
            all_system_content.extend(system_messages)

        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), all_system_content
        )

        # Remove system messages from messages list
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]

        # Add user instruction for JSON output
        new_kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    "Please respond with valid JSON that matches the provided schema. "
                    "Wrap your response in ```json and ``` tags."
                ),
            }
        )

        return response_model, new_kwargs

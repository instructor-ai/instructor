"""Anthropic-specific response handlers."""

import json
from textwrap import dedent
from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.registry import handler_registry
from instructor.utils import combine_system_messages, extract_system_messages

T = TypeVar("T")


class AnthropicToolsHandler(BaseHandler):
    """Handler for Anthropic tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Anthropic tools mode."""
        tool_descriptions = response_model.anthropic_schema
        kwargs["tools"] = [tool_descriptions]
        kwargs["tool_choice"] = {
            "type": "tool",
            "name": response_model.__name__,
        }

        system_messages = extract_system_messages(kwargs.get("messages", []))

        if system_messages:
            kwargs["system"] = combine_system_messages(
                kwargs.get("system"), system_messages
            )

        kwargs["messages"] = [
            m for m in kwargs.get("messages", []) if m["role"] != "system"
        ]

        return response_model, kwargs


class AnthropicReasoningToolsHandler(AnthropicToolsHandler):
    """Handler for Anthropic reasoning tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Anthropic reasoning tools mode."""
        # First apply regular tools handling
        response_model, kwargs = super().handle(response_model, kwargs)

        # Reasoning does not allow forced tool use
        kwargs["tool_choice"] = {"type": "auto"}

        # Add a message recommending only to use the tools if they are relevant
        implicit_forced_tool_message = dedent(
            """
            Return only the tool call and no additional text.
            """
        )
        kwargs["system"] = combine_system_messages(
            kwargs.get("system"),
            [{"type": "text", "text": implicit_forced_tool_message}],
        )
        return response_model, kwargs


class AnthropicJSONHandler(BaseHandler):
    """Handler for Anthropic JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Anthropic JSON mode."""
        system_messages = extract_system_messages(kwargs.get("messages", []))

        if system_messages:
            kwargs["system"] = combine_system_messages(
                kwargs.get("system"), system_messages
            )

        kwargs["messages"] = [
            m for m in kwargs.get("messages", []) if m["role"] != "system"
        ]

        json_schema_message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        kwargs["system"] = combine_system_messages(
            kwargs.get("system"),
            [{"type": "text", "text": json_schema_message}],
        )

        return response_model, kwargs


def register_anthropic_handlers() -> None:
    """Register all Anthropic handlers."""
    handler_registry.register(Mode.ANTHROPIC_TOOLS, AnthropicToolsHandler())
    handler_registry.register(
        Mode.ANTHROPIC_REASONING_TOOLS, AnthropicReasoningToolsHandler()
    )
    handler_registry.register(Mode.ANTHROPIC_JSON, AnthropicJSONHandler())

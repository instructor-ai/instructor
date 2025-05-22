"""Base classes and utilities for response handlers."""

from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class BaseHandler(ABC):
    """Abstract base class for response handlers."""

    @abstractmethod
    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle the response model and kwargs.

        Args:
            response_model: The response model type
            kwargs: Additional keyword arguments

        Returns:
            Tuple of processed response model and updated kwargs
        """
        pass


class ToolsHandler(BaseHandler):
    """Base handler for tools-based modes."""

    def create_tool_definition(self, response_model: type[BaseModel]) -> dict[str, Any]:
        """Create a tool definition from a response model.

        Args:
            response_model: The response model to create a tool from

        Returns:
            Tool definition dictionary
        """
        # Handle different ways openai_schema might be available
        if hasattr(response_model, "openai_schema"):
            schema = response_model.openai_schema
            # If it's a property descriptor, it needs an instance
            if isinstance(schema, property):
                # For properties, we need to check if it's a class attribute
                # In the real code, openai_schema should be a class attribute
                # This is a fallback for test mocks
                schema = {
                    "name": response_model.__name__,
                    "description": response_model.__doc__
                    or f"Schema for {response_model.__name__}",
                    "parameters": response_model.model_json_schema(),
                }
            elif callable(schema):
                schema = schema()
        else:
            # If no openai_schema, generate a basic one
            schema = {
                "name": response_model.__name__,
                "description": response_model.__doc__
                or f"Schema for {response_model.__name__}",
                "parameters": response_model.model_json_schema(),
            }

        return {
            "type": "function",
            "function": schema,
        }

    def set_tool_choice(self, kwargs: dict[str, Any], tool_name: str) -> dict[str, Any]:
        """Set the tool choice in kwargs.

        Args:
            kwargs: The kwargs to update
            tool_name: The name of the tool to choose

        Returns:
            Updated kwargs
        """
        kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_name},
        }
        return kwargs


class JSONHandler(BaseHandler):
    """Base handler for JSON-based modes."""

    def create_json_instruction(self, response_model: type[BaseModel]) -> str:
        """Create a JSON instruction message.

        Args:
            response_model: The response model to create instructions for

        Returns:
            JSON instruction string
        """
        import json

        return dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

    def add_json_instruction_to_messages(
        self, kwargs: dict[str, Any], instruction: str
    ) -> dict[str, Any]:
        """Add JSON instruction to messages.

        Args:
            kwargs: The kwargs containing messages
            instruction: The instruction to add

        Returns:
            Updated kwargs
        """
        messages = kwargs.get("messages", [])

        if not messages or messages[0]["role"] != "system":
            kwargs["messages"] = [{"role": "system", "content": instruction}] + messages
        else:
            # Append to existing system message
            if isinstance(messages[0]["content"], str):
                messages[0]["content"] += f"\n\n{instruction}"
            elif isinstance(messages[0]["content"], list):
                messages[0]["content"][0]["text"] += f"\n\n{instruction}"
            else:
                raise ValueError(
                    "Invalid message format, must be a string or a list of messages"
                )

        return kwargs


class MessageProcessor:
    """Utility class for processing messages."""

    @staticmethod
    def extract_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract system messages from a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            List of system messages
        """
        system_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str):
                    system_messages.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    system_messages.extend(content)
        return system_messages

    @staticmethod
    def combine_system_messages(
        existing: Optional[Any], new_messages: list[dict[str, Any]]
    ) -> Any:
        """Combine existing system messages with new ones.

        Args:
            existing: Existing system message(s)
            new_messages: New system messages to add

        Returns:
            Combined system messages
        """
        if not new_messages:
            return existing

        if existing is None:
            return new_messages if len(new_messages) > 1 else new_messages[0]["text"]

        if isinstance(existing, str):
            combined = [{"type": "text", "text": existing}]
            combined.extend(new_messages)
            return combined
        elif isinstance(existing, list):
            return existing + new_messages
        else:
            return existing

    @staticmethod
    def merge_consecutive_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive messages with the same role.

        Args:
            messages: List of messages to merge

        Returns:
            List of merged messages
        """
        if not messages:
            return messages

        merged = []
        current_role = None
        current_content = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role != current_role:
                if current_role is not None:
                    merged.append(
                        {"role": current_role, "content": "\n".join(current_content)}
                    )
                current_role = role
                current_content = [content] if content else []
            else:
                if content:
                    current_content.append(content)

        if current_role is not None:
            merged.append({"role": current_role, "content": "\n".join(current_content)})

        return merged

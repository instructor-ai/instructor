"""Tests for base handler classes."""

import pytest
from typing import Any, ClassVar

from pydantic import BaseModel

from instructor.response_processing.base import (
    JSONHandler,
    MessageProcessor,
    ToolsHandler,
)


class MockModel(BaseModel):
    """Mock model for testing."""

    name: str
    value: int

    openai_schema: ClassVar[dict[str, Any]] = {
        "name": "MockModel",
        "description": "A mock model",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
        },
    }


class ConcreteToolsHandler(ToolsHandler):
    """Concrete implementation for testing."""

    def handle(self, response_model, kwargs):
        """Concrete handle implementation."""
        return response_model, kwargs


class TestToolsHandler:
    """Test the ToolsHandler class."""

    def test_create_tool_definition(self):
        """Test creating a tool definition."""
        handler = ConcreteToolsHandler()
        tool_def = handler.create_tool_definition(MockModel)

        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "MockModel"
        assert "parameters" in tool_def["function"]

    def test_set_tool_choice(self):
        """Test setting tool choice."""
        handler = ConcreteToolsHandler()
        kwargs = {"test": "value"}

        result = handler.set_tool_choice(kwargs, "test_tool")

        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "test_tool"
        assert result["test"] == "value"

    def test_handle_concrete_works(self):
        """Test that concrete handler works."""
        handler = ConcreteToolsHandler()

        result = handler.handle(MockModel, {"test": "value"})
        assert result == (MockModel, {"test": "value"})


class ConcreteJSONHandler(JSONHandler):
    """Concrete implementation for testing."""

    def handle(self, response_model, kwargs):
        """Concrete handle implementation."""
        return response_model, kwargs


class TestJSONHandler:
    """Test the JSONHandler class."""

    def test_create_json_instruction(self):
        """Test creating JSON instruction."""
        handler = ConcreteJSONHandler()
        instruction = handler.create_json_instruction(MockModel)

        assert "json_schema" in instruction
        assert "MockModel" in str(MockModel.model_json_schema())
        assert "return an instance of the JSON" in instruction

    def test_add_json_instruction_no_messages(self):
        """Test adding JSON instruction when no messages exist."""
        handler = ConcreteJSONHandler()
        kwargs = {"messages": []}
        instruction = "Test instruction"

        result = handler.add_json_instruction_to_messages(kwargs, instruction)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == instruction

    def test_add_json_instruction_no_system_message(self):
        """Test adding JSON instruction when no system message exists."""
        handler = ConcreteJSONHandler()
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        instruction = "Test instruction"

        result = handler.add_json_instruction_to_messages(kwargs, instruction)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == instruction
        assert result["messages"][1]["role"] == "user"

    def test_add_json_instruction_existing_system_string(self):
        """Test adding JSON instruction to existing system message (string)."""
        handler = ConcreteJSONHandler()
        kwargs = {"messages": [{"role": "system", "content": "Existing"}]}
        instruction = "Test instruction"

        result = handler.add_json_instruction_to_messages(kwargs, instruction)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"
        assert "Existing" in result["messages"][0]["content"]
        assert "Test instruction" in result["messages"][0]["content"]

    def test_add_json_instruction_existing_system_list(self):
        """Test adding JSON instruction to existing system message (list)."""
        handler = ConcreteJSONHandler()
        kwargs = {"messages": [{"role": "system", "content": [{"text": "Existing"}]}]}
        instruction = "Test instruction"

        result = handler.add_json_instruction_to_messages(kwargs, instruction)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"
        assert "Existing" in result["messages"][0]["content"][0]["text"]
        assert "Test instruction" in result["messages"][0]["content"][0]["text"]

    def test_add_json_instruction_invalid_format(self):
        """Test adding JSON instruction with invalid message format."""
        handler = ConcreteJSONHandler()
        kwargs = {"messages": [{"role": "system", "content": 123}]}
        instruction = "Test instruction"

        with pytest.raises(ValueError, match="Invalid message format"):
            handler.add_json_instruction_to_messages(kwargs, instruction)

    def test_handle_concrete_works(self):
        """Test that concrete handler works."""
        handler = ConcreteJSONHandler()

        result = handler.handle(MockModel, {"test": "value"})
        assert result == (MockModel, {"test": "value"})


class TestMessageProcessor:
    """Test the MessageProcessor class."""

    def test_extract_system_messages_string(self):
        """Test extracting system messages with string content."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]

        result = MessageProcessor.extract_system_messages(messages)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "System message"

    def test_extract_system_messages_list(self):
        """Test extracting system messages with list content."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Message 1"}]},
            {"role": "user", "content": "User message"},
        ]

        result = MessageProcessor.extract_system_messages(messages)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Message 1"

    def test_extract_system_messages_no_system(self):
        """Test extracting system messages when none exist."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
        ]

        result = MessageProcessor.extract_system_messages(messages)
        assert result == []

    def test_combine_system_messages_none_existing(self):
        """Test combining system messages when existing is None."""
        new_messages = [{"type": "text", "text": "New message"}]

        result = MessageProcessor.combine_system_messages(None, new_messages)
        assert result == "New message"

    def test_combine_system_messages_string_existing(self):
        """Test combining system messages when existing is a string."""
        existing = "Existing message"
        new_messages = [{"type": "text", "text": "New message"}]

        result = MessageProcessor.combine_system_messages(existing, new_messages)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["text"] == "Existing message"
        assert result[1]["text"] == "New message"

    def test_combine_system_messages_list_existing(self):
        """Test combining system messages when existing is a list."""
        existing = [{"type": "text", "text": "Existing"}]
        new_messages = [{"type": "text", "text": "New"}]

        result = MessageProcessor.combine_system_messages(existing, new_messages)

        assert len(result) == 2
        assert result[0]["text"] == "Existing"
        assert result[1]["text"] == "New"

    def test_combine_system_messages_empty_new(self):
        """Test combining when new messages is empty."""
        existing = "Existing"
        new_messages = []

        result = MessageProcessor.combine_system_messages(existing, new_messages)
        assert result == "Existing"

    def test_merge_consecutive_messages_empty(self):
        """Test merging empty message list."""
        result = MessageProcessor.merge_consecutive_messages([])
        assert result == []

    def test_merge_consecutive_messages_single(self):
        """Test merging single message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = MessageProcessor.merge_consecutive_messages(messages)

        assert len(result) == 1
        assert result[0] == messages[0]

    def test_merge_consecutive_messages_different_roles(self):
        """Test merging messages with different roles."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

        result = MessageProcessor.merge_consecutive_messages(messages)
        assert len(result) == 3

    def test_merge_consecutive_messages_same_role(self):
        """Test merging consecutive messages with same role."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine"},
            {"role": "assistant", "content": "Thanks for asking"},
        ]

        result = MessageProcessor.merge_consecutive_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello\nHow are you?"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "I'm fine\nThanks for asking"

    def test_merge_consecutive_messages_empty_content(self):
        """Test merging messages with empty content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "World"},
        ]

        result = MessageProcessor.merge_consecutive_messages(messages)

        assert len(result) == 1
        assert result[0]["content"] == "Hello\nWorld"

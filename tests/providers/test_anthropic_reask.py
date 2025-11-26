"""Tests for Anthropic reask (retry) logic.

This module tests the reask_anthropic_tools function to ensure it properly
handles validation errors by generating correct tool_result messages.
"""

from typing import Any

from anthropic.types import Message, Usage, ToolUseBlock, TextBlock

from instructor.providers.anthropic.utils import reask_anthropic_tools


def _build_tool_use_message(
    tool_use_id: str = "toolu_01ABC123",
    tool_name: str = "TestModel",
    tool_input: dict[str, Any] | None = None,
) -> Message:
    """Build a mock Anthropic Message with a tool_use block."""
    if tool_input is None:
        tool_input = {"name": "invalid", "age": -5}

    return Message(
        id="msg_test_id",
        content=[
            ToolUseBlock(
                type="tool_use",
                id=tool_use_id,
                name=tool_name,
                input=tool_input,
            )
        ],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def _build_text_and_tool_use_message(
    tool_use_id: str = "toolu_01ABC123",
    tool_name: str = "TestModel",
    tool_input: dict[str, Any] | None = None,
    text_content: str = "I'll help you with that.",
) -> Message:
    """Build a mock Anthropic Message with both text and tool_use blocks."""
    if tool_input is None:
        tool_input = {"name": "invalid", "age": -5}

    return Message(
        id="msg_test_id",
        content=[
            TextBlock(type="text", text=text_content),
            ToolUseBlock(
                type="tool_use",
                id=tool_use_id,
                name=tool_name,
                input=tool_input,
            ),
        ],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def _build_multi_tool_use_message() -> Message:
    """Build a mock Anthropic Message with multiple tool_use blocks."""
    return Message(
        id="msg_test_id",
        content=[
            ToolUseBlock(
                type="tool_use",
                id="toolu_01FIRST",
                name="FirstModel",
                input={"field1": "value1"},
            ),
            ToolUseBlock(
                type="tool_use",
                id="toolu_02SECOND",
                name="SecondModel",
                input={"field2": "value2"},
            ),
        ],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def test_reask_adds_tool_result_for_single_tool_use() -> None:
    """Verify that reask adds a proper tool_result block after tool_use."""
    kwargs: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "Extract user info from: John, 25 years old",
            }
        ]
    }
    response = _build_tool_use_message(
        tool_use_id="toolu_01ABC123",
        tool_input={"name": "John", "age": -5},
    )
    exception = ValueError("age must be positive")

    result = reask_anthropic_tools(kwargs, response, exception)

    # Should have 3 messages: original user, assistant with tool_use, user with tool_result
    assert len(result["messages"]) == 3

    # Second message should be assistant with tool_use block
    assistant_msg = result["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["content"]) == 1
    assert assistant_msg["content"][0]["type"] == "tool_use"
    assert assistant_msg["content"][0]["id"] == "toolu_01ABC123"

    # Third message should be user with tool_result block
    user_msg = result["messages"][2]
    assert user_msg["role"] == "user"
    assert len(user_msg["content"]) == 1
    assert user_msg["content"][0]["type"] == "tool_result"
    assert user_msg["content"][0]["tool_use_id"] == "toolu_01ABC123"
    assert user_msg["content"][0]["is_error"] is True
    assert "Validation Error" in user_msg["content"][0]["content"]


def test_reask_handles_text_and_tool_use_blocks() -> None:
    """Verify reask works when response has both text and tool_use blocks."""
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Extract user info"}]
    }
    response = _build_text_and_tool_use_message(
        tool_use_id="toolu_01XYZ",
        text_content="Let me extract that for you.",
    )
    exception = ValueError("validation failed")

    result = reask_anthropic_tools(kwargs, response, exception)

    # Assistant message should preserve both text and tool_use blocks
    assistant_msg = result["messages"][1]
    assert len(assistant_msg["content"]) == 2
    assert assistant_msg["content"][0]["type"] == "text"
    assert assistant_msg["content"][1]["type"] == "tool_use"

    # User message should have tool_result for the tool_use
    user_msg = result["messages"][2]
    assert user_msg["content"][0]["type"] == "tool_result"
    assert user_msg["content"][0]["tool_use_id"] == "toolu_01XYZ"


def test_reask_handles_multiple_tool_use_blocks() -> None:
    """Verify reask handles responses with multiple tool_use blocks.

    According to Anthropic's API, each tool_use block must have a
    corresponding tool_result block. This test ensures all tool_use
    blocks get proper tool_result responses.

    This is the bug reported in issue #1938 - when there are multiple
    tool_use blocks, only the last one gets a tool_result.
    """
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Extract multiple items"}]
    }
    response = _build_multi_tool_use_message()
    exception = ValueError("validation failed on first model")

    result = reask_anthropic_tools(kwargs, response, exception)

    # Get the user message with tool_result(s)
    user_msg = result["messages"][2]
    assert user_msg["role"] == "user"

    # Extract all tool_use IDs from assistant message
    assistant_msg = result["messages"][1]
    tool_use_ids = [
        block["id"]
        for block in assistant_msg["content"]
        if block.get("type") == "tool_use"
    ]

    # Extract all tool_result IDs from user message
    tool_result_ids = [
        block["tool_use_id"]
        for block in user_msg["content"]
        if block.get("type") == "tool_result"
    ]

    # CRITICAL: Every tool_use must have a corresponding tool_result
    # This is the requirement from Anthropic's API
    assert set(tool_use_ids) == set(tool_result_ids), (
        f"Missing tool_result blocks. "
        f"tool_use IDs: {tool_use_ids}, tool_result IDs: {tool_result_ids}"
    )


def test_reask_does_not_modify_original_kwargs() -> None:
    """Verify that reask_anthropic_tools does not modify the original kwargs."""
    original_messages: list[dict[str, str]] = [
        {"role": "user", "content": "Test message"}
    ]
    kwargs: dict[str, Any] = {"messages": original_messages.copy()}
    response = _build_tool_use_message()
    exception = ValueError("test error")

    result = reask_anthropic_tools(kwargs, response, exception)

    # Original should be unchanged
    assert len(kwargs["messages"]) == 1
    # Result should have more messages
    assert len(result["messages"]) == 3


def test_reask_without_tool_use_adds_user_message() -> None:
    """Verify reask handles responses with no tool_use blocks."""
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Extract user info"}]
    }
    # Create a response with only text, no tool_use
    response = Message(
        id="msg_test_id",
        content=[TextBlock(type="text", text="I couldn't use the tool.")],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )
    exception = ValueError("no tool was called")

    result = reask_anthropic_tools(kwargs, response, exception)

    # Should have 3 messages
    assert len(result["messages"]) == 3

    # The user message should be a regular content string (not tool_result)
    user_msg = result["messages"][2]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], str)
    assert "no tool invocation" in user_msg["content"]

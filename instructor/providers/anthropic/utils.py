"""Anthropic-specific utilities.

This module contains utilities specific to the Anthropic provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, TypedDict, Union


from ...mode import Mode
from ...processing.schema import generate_anthropic_schema


class SystemMessage(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict[str, str]


def combine_system_messages(
    existing_system: Union[str, list[SystemMessage], None],  # noqa: UP007
    new_system: Union[str, list[SystemMessage]],  # noqa: UP007
) -> Union[str, list[SystemMessage]]:  # noqa: UP007
    """
    Combine existing and new system messages.

    This optimized version uses a more direct approach with fewer branches.

    Args:
        existing_system: Existing system message(s) or None
        new_system: New system message(s) to add

    Returns:
        Combined system message(s)
    """
    # Fast path for None existing_system (avoid unnecessary operations)
    if existing_system is None:
        return new_system

    # Validate input types
    if not isinstance(existing_system, (str, list)) or not isinstance(
        new_system, (str, list)
    ):
        raise ValueError(
            f"System messages must be strings or lists, got {type(existing_system)} and {type(new_system)}"
        )

    # Use direct type comparison instead of isinstance for better performance
    if isinstance(existing_system, str) and isinstance(new_system, str):
        # Both are strings, join with newlines
        # Avoid creating intermediate strings by joining only once
        return f"{existing_system}\n\n{new_system}"
    elif isinstance(existing_system, list) and isinstance(new_system, list):
        # Both are lists, use list extension in place to avoid creating intermediate lists
        # First create a new list to avoid modifying the original
        result = list(existing_system)
        result.extend(new_system)
        return result
    elif isinstance(existing_system, str) and isinstance(new_system, list):
        # existing is string, new is list
        # Create a pre-sized list to avoid resizing
        result = [SystemMessage(type="text", text=existing_system)]
        result.extend(new_system)
        return result
    elif isinstance(existing_system, list) and isinstance(new_system, str):
        # existing is list, new is string
        # Create message once and add to existing
        new_message = SystemMessage(type="text", text=new_system)
        result = list(existing_system)
        result.append(new_message)
        return result

    # This should never happen due to validation above
    return existing_system


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    """
    Extract system messages from a list of messages.

    This optimized version pre-allocates the result list and
    reduces function call overhead.

    Args:
        messages: List of messages to extract system messages from

    Returns:
        List of system messages
    """
    # Fast path for empty messages
    if not messages:
        return []

    # First count system messages to pre-allocate result list
    system_count = sum(1 for m in messages if m.get("role") == "system")

    # If no system messages, return empty list
    if system_count == 0:
        return []

    # Helper function to convert a message content to SystemMessage
    def convert_message(content: Any) -> SystemMessage:
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        elif isinstance(content, dict):
            return SystemMessage(**content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    # Process system messages
    result: list[SystemMessage] = []

    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")

            # Skip empty content
            if not content:
                continue

            # Handle list or single content
            if isinstance(content, list):
                # Process each item in the list
                for item in content:
                    if item:  # Skip empty items
                        result.append(convert_message(item))
            else:
                # Process single content
                result.append(convert_message(content))

    return result


def reask_anthropic_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Anthropic tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (tool result messages indicating validation errors)
    """
    kwargs = kwargs.copy()
    from anthropic.types import Message

    # Handle Stream objects which are not Message instances
    # This happens when streaming mode is used with retries
    if not isinstance(response, Message):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )
        return kwargs

    assistant_content = []
    tool_use_id = None
    for content in response.content:
        assistant_content.append(content.model_dump())  # type: ignore
        if content.type == "tool_use":
            tool_use_id = content.id

    reask_msgs = [{"role": "assistant", "content": assistant_content}]  # type: ignore
    if tool_use_id is not None:
        reask_msgs.append(  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
                        "is_error": True,
                    }
                ],
            }
        )
    else:
        reask_msgs.append(  # type: ignore
            {
                "role": "user",
                "content": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors",
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_anthropic_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Anthropic JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    from anthropic.types import Message

    # Handle Stream objects which are not Message instances
    # This happens when streaming mode is used with retries
    if not isinstance(response, Message):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Validation Errors found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )
        return kwargs

    # Filter for text blocks to handle ThinkingBlock and other non-text content
    text_blocks = [c for c in response.content if c.type == "text"]
    if not text_blocks:
        # Fallback if no text blocks found
        text_content = "No text content found in response"
    else:
        # Use the last text block, similar to function_calls.py:396-397
        text_content = text_blocks[-1].text

    reask_msg = {
        "role": "user",
        "content": f"""Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{text_content}""",
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def handle_anthropic_message_conversion(new_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Handle message conversion for Anthropic modes when response_model is None.

    Kwargs modifications:
    - Modifies: "messages" (removes system messages)
    - Adds/Modifies: "system" (if system messages found in messages)
    """
    messages = new_kwargs.get("messages", [])

    # Handle Anthropic style messages
    new_kwargs["messages"] = [m for m in messages if m["role"] != "system"]

    if "system" not in new_kwargs:
        system_messages = extract_system_messages(messages)
        if system_messages:
            new_kwargs["system"] = system_messages

    return new_kwargs


def handle_anthropic_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Anthropic tools mode.

    When response_model is None:
        - Extracts system messages from the messages list and moves them to the 'system' parameter
        - Filters out system messages from the messages list
        - No tools are configured
        - Allows for unstructured responses from Claude

    When response_model is provided:
        - Generates Anthropic tool schema from the response model
        - Sets up forced tool use with the specific tool name
        - Extracts and combines system messages
        - Filters system messages from the messages list

    Kwargs modifications:
    - Modifies: "messages" (removes system messages)
    - Adds/Modifies: "system" (combines existing with extracted system messages)
    - Adds: "tools" (list with tool schema) - only when response_model provided
    - Adds: "tool_choice" (forced tool use) - only when response_model provided
    """
    if response_model is None:
        # Just handle message conversion
        new_kwargs = handle_anthropic_message_conversion(new_kwargs)
        return None, new_kwargs

    tool_descriptions = generate_anthropic_schema(response_model)
    new_kwargs["tools"] = [tool_descriptions]
    new_kwargs["tool_choice"] = {
        "type": "tool",
        "name": response_model.__name__,
    }

    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    return response_model, new_kwargs


def handle_anthropic_reasoning_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Anthropic reasoning tools mode.

    This mode is similar to regular tools mode but with reasoning enabled:
    - Uses "auto" tool choice instead of forced tool use
    - Adds a system message encouraging tool use only when relevant
    - Allows Claude to reason about whether to use tools

    When response_model is None:
        - Performs the same message conversion as handle_anthropic_tools
        - No tools are configured

    When response_model is provided:
        - Sets up tools as in regular tools mode
        - Changes tool_choice to "auto" to allow reasoning
        - Adds system message to guide tool usage

    Kwargs modifications:
    - All modifications from handle_anthropic_tools, plus:
    - Modifies: "tool_choice" (changes to {"type": "auto"}) - only when response_model provided
    - Modifies: "system" (adds implicit forced tool message)
    """
    # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#forcing-tool-use

    response_model, new_kwargs = handle_anthropic_tools(response_model, new_kwargs)

    if response_model is None:
        # Just handle message conversion - already done by handle_anthropic_tools
        return None, new_kwargs

    # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#forcing-tool-use
    # Reasoning does not allow forced tool use
    new_kwargs["tool_choice"] = {"type": "auto"}

    # But add a message recommending only to use the tools if they are relevant
    implict_forced_tool_message = dedent(
        f"""
        Return only the tool call and no additional text.
        """
    )
    new_kwargs["system"] = combine_system_messages(
        new_kwargs.get("system"),
        [{"type": "text", "text": implict_forced_tool_message}],
    )
    return response_model, new_kwargs


def handle_anthropic_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Anthropic JSON mode.

    This mode instructs Claude to return JSON responses:
    - System messages are extracted and combined
    - A JSON schema message is added to guide the response format

    When response_model is None:
        - Extracts and moves system messages to the 'system' parameter
        - Filters system messages from the messages list
        - No JSON schema is added

    When response_model is provided:
        - Performs system message handling as above
        - Adds a system message with the JSON schema
        - Instructs Claude to return an instance matching the schema

    Kwargs modifications:
    - Modifies: "messages" (removes system messages)
    - Adds/Modifies: "system" (combines existing with extracted system messages)
    - Modifies: "system" (adds JSON schema message) - only when response_model provided
    """
    import json

    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    if response_model is None:
        # Just handle message conversion - already done above
        return None, new_kwargs

    json_schema_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    new_kwargs["system"] = combine_system_messages(
        new_kwargs.get("system"),
        [{"type": "text", "text": json_schema_message}],
    )

    return response_model, new_kwargs


def handle_anthropic_parallel_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """
    Handle Anthropic parallel tools mode.

    Kwargs modifications:
    - Adds: "tools" (multiple function schemas from parallel model)
    - Adds: "tool_choice" ("auto" to allow model to choose which tools to call)
    - Modifies: "system" (moves system messages into system parameter)
    - Removes: "system" messages from "messages" list
    - Validates: stream=False
    """
    from ...dsl.parallel import (
        AnthropicParallelModel,
        handle_anthropic_parallel_model,
    )
    from ...core.exceptions import ConfigurationError

    if new_kwargs.get("stream", False):
        raise ConfigurationError(
            "stream=True is not supported when using ANTHROPIC_PARALLEL_TOOLS mode"
        )

    new_kwargs["tools"] = handle_anthropic_parallel_model(response_model)
    new_kwargs["tool_choice"] = {"type": "auto"}

    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    return AnthropicParallelModel(typehint=response_model), new_kwargs


# Handler registry for Anthropic
ANTHROPIC_HANDLERS = {
    Mode.ANTHROPIC_TOOLS: {
        "reask": reask_anthropic_tools,
        "response": handle_anthropic_tools,
    },
    Mode.ANTHROPIC_JSON: {
        "reask": reask_anthropic_json,
        "response": handle_anthropic_json,
    },
    Mode.ANTHROPIC_REASONING_TOOLS: {
        "reask": reask_anthropic_tools,
        "response": handle_anthropic_reasoning_tools,
    },
    Mode.ANTHROPIC_PARALLEL_TOOLS: {
        "reask": reask_anthropic_tools,
        "response": handle_anthropic_parallel_tools,
    },
}

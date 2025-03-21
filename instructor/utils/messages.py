import json
from typing import Any, Callable

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if (
        hasattr(message, "function_call")
        and message.function_call is not None
        and ret["content"]
    ):
        if not isinstance(ret["content"], str):
            response_message: str = ""
            for content_message in ret["content"]:
                if "text" in content_message:
                    response_message += content_message["text"]
                elif "refusal" in content_message:
                    response_message += content_message["refusal"]
            ret["content"] = response_message
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge consecutive messages from the same role into a single message.

    This optimized version pre-allocates the result list and minimizes operations.

    Args:
        messages: List of message dictionaries to merge

    Returns:
        List of merged message dictionaries
    """
    if not messages:
        return []

    # Pre-allocate result list with estimated size (worst case: no merges happen)
    message_count = len(messages)
    new_messages = []

    # Detect whether all messages have a flat content (i.e. all string)
    # Some providers require content to be a string, so we need to check that and behave accordingly
    # Fast path: avoid checking all messages if the first few have mixed content types
    flat_string = True
    for _i, m in enumerate(messages[: min(10, message_count)]):
        if not isinstance(m.get("content", ""), str):
            flat_string = False
            break

    # Only check all messages if we haven't determined it's not flat_string
    if flat_string and message_count > 10:
        flat_string = all(isinstance(m.get("content", ""), str) for m in messages[10:])

    # Process messages with a single loop
    for message in messages:
        role = message.get("role", "user")
        new_content = message.get("content", "")

        # Transform string content to list if needed
        if not flat_string and isinstance(new_content, str):
            new_content = [{"type": "text", "text": new_content}]

        # Check if we can merge with previous message
        if new_messages and role == new_messages[-1]["role"]:
            if flat_string:
                # Fast path for string content
                new_messages[-1]["content"] += f"\n\n{new_content}"
            else:
                # Fast path for list content
                if isinstance(new_content, list):
                    new_messages[-1]["content"].extend(new_content)
                else:
                    # Fallback for unexpected content type
                    new_messages[-1]["content"].append(new_content)
        else:
            # Add new message
            new_messages.append({"role": role, "content": new_content})

    return new_messages


def get_message_content(message: ChatCompletionMessageParam) -> list[Any]:
    """
    Extract content from a message and ensure it's returned as a list.

    This optimized version handles different message formats more efficiently.

    Args:
        message: A message in ChatCompletionMessageParam format

    Returns:
        The message content as a list
    """
    # Fast path for empty message
    if not message:
        return [""]

    # Get content with default empty string
    content = message.get("content", "")

    # Fast path for common content types
    if isinstance(content, list):
        return content if content else [""]

    # Return single item list with content (could be string, None, or other)
    return [content if content is not None else ""]


def is_async(func: Callable[..., Any]) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    import inspect

    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__  # type: ignore - dynamic
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine

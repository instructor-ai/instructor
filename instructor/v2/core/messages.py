"""Message helpers owned by the v2 runtime."""

from __future__ import annotations

import json
from typing import Any

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam


def extract_messages(kwargs: dict[str, Any]) -> Any:
    if "messages" in kwargs:
        return kwargs["messages"]
    if "contents" in kwargs:
        return kwargs["contents"]
    if "chat_history" in kwargs:
        return kwargs["chat_history"]
    return []


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
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
            response_message = ""
            for content_message in ret["content"]:
                if isinstance(content_message, dict):
                    if content_message.get("type") == "text":
                        response_message += content_message.get("text", "")
                    elif content_message.get("type") == "refusal":
                        response_message += content_message.get("refusal", "")
            ret["content"] = response_message
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []

    message_count = len(messages)
    new_messages: list[dict[str, Any]] = []
    flat_string = True
    for message in messages[: min(10, message_count)]:
        if not isinstance(message.get("content", ""), str):
            flat_string = False
            break

    if flat_string and message_count > 10:
        flat_string = all(
            isinstance(message.get("content", ""), str) for message in messages[10:]
        )

    for message in messages:
        role = message.get("role", "user")
        new_content = message.get("content", "")
        if not flat_string and isinstance(new_content, str):
            new_content = [{"type": "text", "text": new_content}]

        if new_messages and role == new_messages[-1]["role"]:
            if flat_string:
                new_messages[-1]["content"] += f"\n\n{new_content}"
            elif isinstance(new_content, list):
                new_messages[-1]["content"].extend(new_content)
            else:
                new_messages[-1]["content"].append(new_content)
        else:
            new_messages.append({"role": role, "content": new_content})

    return new_messages


def get_message_content(message: ChatCompletionMessageParam) -> list[Any]:
    """Return message content in list form for Gemini-style APIs."""
    if not message:
        return [""]
    content = message.get("content", "")
    if isinstance(content, list):
        return content if content else [""]
    return [content if content is not None else ""]

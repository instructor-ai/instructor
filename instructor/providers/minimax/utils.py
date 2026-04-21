"""MiniMax-specific utilities.

This module contains utilities specific to the MiniMax provider,
including reask functions and response handlers. MiniMax exposes an
OpenAI-compatible chat completions API, so the handlers delegate to the
shared OpenAI helpers.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from ...mode import Mode
from ...processing.schema import generate_openai_schema
from ...utils.core import dump_message


def reask_minimax_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for MiniMax tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (tool response messages indicating validation errors)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_minimax_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for MiniMax JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                "Correct your JSON ONLY RESPONSE, based on the following errors:\n"
                f"{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_minimax_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle MiniMax tools mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "tools" (list with function schema)
      - Adds: "tool_choice" (forced function call)
    """
    if response_model is None:
        return None, new_kwargs

    schema = generate_openai_schema(response_model)
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": schema["name"]},
    }
    return response_model, new_kwargs


def handle_minimax_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle MiniMax JSON mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "response_format" with type="json_object"
      - Appends a user message describing the expected JSON schema
    """
    if response_model is None:
        return None, new_kwargs

    message = dedent(
        f"""
        Parse the content and return a JSON object matching this schema:

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Return a valid JSON instance, not the schema definition."""
    )

    new_kwargs["response_format"] = {"type": "json_object"}
    new_kwargs["messages"].append(
        {
            "role": "user",
            "content": message,
        }
    )
    return response_model, new_kwargs


# Handler registry for MiniMax
MINIMAX_HANDLERS = {
    Mode.MINIMAX_TOOLS: {
        "reask": reask_minimax_tools,
        "response": handle_minimax_tools,
    },
    Mode.MINIMAX_JSON: {
        "reask": reask_minimax_json,
        "response": handle_minimax_json,
    },
}

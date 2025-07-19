"""Fireworks-specific utilities.

This module contains utilities specific to the Fireworks provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
from ...processing.schema import generate_openai_schema
from ...utils.core import dump_message


def reask_fireworks_tools(kwargs: dict[str, Any], response: Any, exception: Exception):
    """
    Handle reask for Fireworks tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (tool response messages indicating validation errors)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",  # type: ignore
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_fireworks_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Fireworks JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_fireworks_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Fireworks tools mode.

    Kwargs modifications:
    - Adds: "tools" (list with function schema)
    - Adds: "tool_choice" (forced function call)
    - Sets default: stream=False
    """
    if "stream" not in new_kwargs:
        new_kwargs["stream"] = False
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": generate_openai_schema(response_model),
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": generate_openai_schema(response_model)["name"]},
    }
    return response_model, new_kwargs


def handle_fireworks_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Fireworks JSON mode.

    Kwargs modifications:
    - Adds: "response_format" with json_schema
    - Sets default: stream=False
    """
    if "stream" not in new_kwargs:
        new_kwargs["stream"] = False

    new_kwargs["response_format"] = {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
    return response_model, new_kwargs


# Handler registry for Fireworks
FIREWORKS_HANDLERS = {
    Mode.FIREWORKS_TOOLS: {
        "reask": reask_fireworks_tools,
        "response": handle_fireworks_tools,
    },
    Mode.FIREWORKS_JSON: {
        "reask": reask_fireworks_json,
        "response": handle_fireworks_json,
    },
}

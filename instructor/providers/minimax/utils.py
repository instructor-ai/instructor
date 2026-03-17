"""MiniMax-specific utilities.

This module contains utilities specific to the MiniMax provider,
including reask functions, response handlers, and message formatting.

MiniMax uses an OpenAI-compatible API, so the handling is similar
to the standard OpenAI tools and JSON modes.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
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
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors and call the tool {tool_call.function.name} again, "
                    f"taking into account the problems with {tool_call.function.arguments} that was previously generated."
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
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle MiniMax tools mode.

    Uses OpenAI-compatible tool calling format.

    Kwargs modifications:
    - Adds: "tools" and "tool_choice" for function calling
    """
    from ...processing.function_calls import openai_schema

    schema = openai_schema(response_model)
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": schema.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": schema.openai_schema["name"]},
    }
    return response_model, new_kwargs


def handle_minimax_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle MiniMax JSON mode.

    Uses response_format with json_schema for structured output.

    Kwargs modifications:
    - Adds: "response_format" with json_schema type
    """
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
        },
    }
    return response_model, new_kwargs


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

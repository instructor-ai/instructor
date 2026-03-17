"""MiniMax-specific utilities.

This module contains utilities specific to the MiniMax provider,
including reask functions, response handlers, and message formatting.

MiniMax uses an OpenAI-compatible API but does not support
``response_format``, so JSON mode uses system-prompt guidance instead.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
from ...utils.core import dump_message
from ...processing.schema import generate_openai_schema


def reask_minimax_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for MiniMax tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (tool response messages indicating validation errors)
    """
    kwargs = kwargs.copy()

    if response is None or not hasattr(response, "choices"):
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

    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors and call the tool {tool_call.function.name} again, "
                    f"taking into account the problems with {tool_call.function.arguments} "
                    f"that was previously generated."
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
    """Handle reask for MiniMax JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_minimax_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle MiniMax tools mode.

    Kwargs modifications:
    - Adds: "tools" (list with function schema)
    - Adds: "tool_choice" (forced function call)
    """
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


def handle_minimax_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle MiniMax JSON mode.

    MiniMax does not support ``response_format``, so we instruct the model
    to return valid JSON via the system prompt.

    Kwargs modifications:
    - Adds: "messages" (system instruction with JSON schema)
    """
    instruction = (
        "You must respond with valid JSON only. No markdown, no explanation, "
        "no additional text. Only output a JSON object.\n\n"
        f"Here is the JSON schema to follow:\n\n"
        f"{response_model.model_json_schema()}\n\n"
        f"Your response should be a valid JSON object that "
        f"`{response_model.__name__}.model_validate_json()` can parse."
    )

    new_kwargs["messages"] = [
        {"role": "system", "content": instruction}
    ] + new_kwargs["messages"]
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

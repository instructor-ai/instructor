"""Mistral-specific utilities.

This module contains utilities specific to the Mistral provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
from ...processing.schema import generate_openai_schema
from ...utils.core import dump_message


def reask_mistral_structured_outputs(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Mistral structured outputs mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (assistant content and user correction request)
    """
    kwargs = kwargs.copy()
    reask_msgs = [
        {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
    ]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_mistral_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Mistral tools mode when validation fails.

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


def handle_mistral_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Mistral tools mode.

    Kwargs modifications:
    - Adds: "tools" (list with function schema)
    - Adds: "tool_choice" set to "any"
    """
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": generate_openai_schema(response_model),
        }
    ]
    new_kwargs["tool_choice"] = "any"
    return response_model, new_kwargs


def handle_mistral_structured_outputs(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Mistral structured outputs mode.

    Kwargs modifications:
    - Adds: "response_format" derived from the response model
    - Removes: "tools" and "response_model" from kwargs
    """
    from mistralai.extra import response_format_from_pydantic_model

    new_kwargs["response_format"] = response_format_from_pydantic_model(response_model)
    new_kwargs.pop("tools", None)
    new_kwargs.pop("response_model", None)
    return response_model, new_kwargs


# Handler registry for Mistral
MISTRAL_HANDLERS = {
    Mode.MISTRAL_TOOLS: {
        "reask": reask_mistral_tools,
        "response": handle_mistral_tools,
    },
    Mode.MISTRAL_STRUCTURED_OUTPUTS: {
        "reask": reask_mistral_structured_outputs,
        "response": handle_mistral_structured_outputs,
    },
}

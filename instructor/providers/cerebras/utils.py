"""Cerebras-specific utilities.

This module contains utilities specific to the Cerebras provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
from ...utils.core import dump_message
from ...processing.schema import generate_openai_schema


def reask_cerebras_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Cerebras tools mode when validation fails.

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


def handle_cerebras_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Cerebras tools mode.

    Kwargs modifications:
    - Adds: "tools" (list with function schema)
    - Adds: "tool_choice" (forced function call)
    - Validates: stream=False
    """
    if new_kwargs.get("stream", False):
        raise ValueError("Stream is not supported for Cerebras Tool Calling")
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


def handle_cerebras_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Cerebras JSON mode.

    Kwargs modifications:
    - Adds: "messages" (system instruction with JSON schema)
    """
    instruction = f"""
You are a helpful assistant that excels at following instructions.Your task is to understand the content and provide the parsed objects in json that match the following json_schema:\n

Here is the relevant JSON schema to adhere to

<schema>
{response_model.model_json_schema()}
</schema>

Your response should consist only of a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""

    new_kwargs["messages"] = [{"role": "system", "content": instruction}] + new_kwargs[
        "messages"
    ]
    return response_model, new_kwargs


# Handler registry for Cerebras
CEREBRAS_HANDLERS = {
    Mode.CEREBRAS_TOOLS: {
        "reask": reask_cerebras_tools,
        "response": handle_cerebras_tools,
    },
    Mode.CEREBRAS_JSON: {
        "reask": reask_cerebras_tools,  # Uses same reask as tools
        "response": handle_cerebras_json,
    },
}

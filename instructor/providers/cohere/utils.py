"""Cohere-specific utilities.

This module contains utilities specific to the Cohere provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode


def reask_cohere_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Cohere
    exception: Exception,
):
    """
    Handle reask for Cohere tools and JSON schema modes.

    Kwargs modifications:
    - Adds/Modifies: "chat_history" (appends prior message)
    - Modifies: "message" (user prompt describing validation errors)
    """
    # Get message outside the function
    message = kwargs.get("message", "")

    # Fetch or initialize chat_history in one operation
    if "chat_history" in kwargs:
        # Only modify chat_history if it exists
        kwargs["chat_history"].append({"role": "user", "message": message})
    else:
        # Create a new chat_history if it doesn't exist
        kwargs["chat_history"] = [{"role": "user", "message": message}]

    # Set the message directly without string concatenation with f-strings
    kwargs["message"] = (
        "Correct the following JSON response, based on the errors given below:\n\n"
        f"JSON:\n{response.text}\n\nExceptions:\n{exception}"
    )
    return kwargs


def handle_cohere_modes(new_kwargs: dict[str, Any]) -> tuple[None, dict[str, Any]]:
    """
    Convert OpenAI-style messages to Cohere format.

    Kwargs modifications:
    - Removes: "messages"
    - Adds: "message" (last user message)
    - Adds: "chat_history" (prior messages)
    - Renames: "model_name" -> "model"
    - Removes: "strict"
    """
    messages = new_kwargs.pop("messages", [])
    chat_history = []
    for message in messages[:-1]:
        chat_history.append(  # type: ignore
            {
                "role": message["role"],
                "message": message["content"],
            }
        )
    new_kwargs["message"] = messages[-1]["content"]
    new_kwargs["chat_history"] = chat_history
    if "model_name" in new_kwargs and "model" not in new_kwargs:
        new_kwargs["model"] = new_kwargs.pop("model_name")
    new_kwargs.pop("strict", None)
    return None, new_kwargs


def handle_cohere_json_schema(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Cohere JSON schema mode.

    When response_model is None:
        - Converts messages from OpenAI format to Cohere format (message + chat_history)
        - No schema is added to the request

    When response_model is provided:
        - Converts messages from OpenAI format to Cohere format
        - Adds the model's JSON schema to response_format

    Kwargs modifications:
    - Removes: "messages" (converted to message + chat_history)
    - Adds: "message" (last message content)
    - Adds: "chat_history" (all messages except last)
    - Modifies: "model" (if "model_name" exists, renames to "model")
    - Removes: "strict"
    - Adds: "response_format" (with JSON schema) - only when response_model provided
    """
    if response_model is None:
        # Just handle message conversion
        return handle_cohere_modes(new_kwargs)

    new_kwargs["response_format"] = {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
    _, new_kwargs = handle_cohere_modes(new_kwargs)

    return response_model, new_kwargs


def handle_cohere_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Cohere tools mode.

    When response_model is None:
        - Converts messages from OpenAI format to Cohere format (message + chat_history)
        - No tools or schema instructions are added
        - Allows for unstructured responses from Cohere

    When response_model is provided:
        - Converts messages from OpenAI format to Cohere format
        - Prepends extraction instructions to the chat history
        - Includes the model's JSON schema in the instructions
        - The model is instructed to extract a valid object matching the schema

    Kwargs modifications:
    - All modifications from handle_cohere_modes (message format conversion)
    - Modifies: "chat_history" (prepends extraction instruction) - only when response_model provided
    """
    if response_model is None:
        # Just handle message conversion
        return handle_cohere_modes(new_kwargs)

    _, new_kwargs = handle_cohere_modes(new_kwargs)

    instruction = f"""\
Extract a valid {response_model.__name__} object based on the chat history and the json schema below.
{response_model.model_json_schema()}
The JSON schema was obtained by running:
```python
schema = {response_model.__name__}.model_json_schema()
```

The output must be a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""
    new_kwargs["chat_history"] = [
        {"role": "user", "message": instruction}
    ] + new_kwargs["chat_history"]
    return response_model, new_kwargs


# Handler registry for Cohere
COHERE_HANDLERS = {
    Mode.COHERE_TOOLS: {
        "reask": reask_cohere_tools,
        "response": handle_cohere_tools,
    },
    Mode.COHERE_JSON_SCHEMA: {
        "reask": reask_cohere_tools,
        "response": handle_cohere_json_schema,
    },
}

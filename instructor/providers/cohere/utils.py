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
    Supports both V1 and V2 formats.

    V1 kwargs modifications:
    - Adds/Modifies: "chat_history" (appends prior message)
    - Modifies: "message" (user prompt describing validation errors)

    V2 kwargs modifications:
    - Modifies: "messages" (appends error correction message)
    """
    # Default to marker stored on kwargs (set during client initialization)
    client_version = kwargs.get("_cohere_client_version")

    # Detect V1 vs V2 response structure and extract text
    if hasattr(response, "text"):
        client_version = "v1"
        response_text = response.text
    elif hasattr(response, "message") and hasattr(response.message, "content"):
        client_version = "v2"
        content_items = response.message.content
        response_text = ""
        if content_items:
            # Find the text content item (skip thinking/other types)
            for item in content_items:
                if (
                    hasattr(item, "type")
                    and item.type == "text"
                    and hasattr(item, "text")
                ):
                    response_text = item.text
                    break
        if not response_text:
            response_text = str(response)
    else:
        # Fallback to string representation
        response_text = str(response)
        if client_version is None:
            if "messages" in kwargs:
                client_version = "v2"
            elif "chat_history" in kwargs or "message" in kwargs:
                client_version = "v1"

    # Create the correction message
    correction_msg = (
        "Correct the following JSON response, based on the errors given below:\n\n"
        f"JSON:\n{response_text}\n\nExceptions:\n{exception}"
    )

    if client_version == "v2":
        # V2 format: append to messages list
        kwargs["messages"].append({"role": "user", "content": correction_msg})
    elif client_version == "v1":
        # V1 format: use chat_history and message
        message = kwargs.get("message", "")

        # Fetch or initialize chat_history in one operation
        if "chat_history" in kwargs:
            kwargs["chat_history"].append({"role": "user", "message": message})
        else:
            kwargs["chat_history"] = [{"role": "user", "message": message}]

        kwargs["message"] = correction_msg
    else:
        # Unknown version - raise error for future compatibility
        raise ValueError(
            f"Unsupported Cohere client version: {client_version}. "
            f"Expected 'v1' or 'v2'."
        )

    return kwargs


def handle_cohere_modes(new_kwargs: dict[str, Any]) -> tuple[None, dict[str, Any]]:
    """
    Convert OpenAI-style messages to Cohere format.
    Handles both V1 and V2 client formats.

    V1 format:
    - Removes: "messages"
    - Adds: "message" (last user message)
    - Adds: "chat_history" (prior messages)

    V2 format:
    - Keeps: "messages" (compatible with OpenAI format)

    Both versions:
    - Renames: "model_name" -> "model"
    - Removes: "strict"
    - Removes: "_cohere_client_version" (internal marker)
    """
    new_kwargs = new_kwargs.copy()
    client_version = new_kwargs.pop("_cohere_client_version")

    if client_version == "v2":
        # V2 uses OpenAI-style messages directly - no conversion needed
        # Just clean up incompatible fields
        if "model_name" in new_kwargs and "model" not in new_kwargs:
            new_kwargs["model"] = new_kwargs.pop("model_name")
        new_kwargs.pop("strict", None)
    elif client_version == "v1":
        # V1 needs conversion from OpenAI format to Cohere V1 format
        messages = new_kwargs.pop("messages", [])
        chat_history = []
        for message in messages[:-1]:
            chat_history.append(  # type: ignore[arg-type]
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
    else:
        # Unknown version - raise error for future compatibility
        raise ValueError(
            f"Unsupported Cohere client version: {client_version}. "
            f"Expected 'v1' or 'v2'."
        )

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
        - Converts messages from OpenAI format to Cohere format (message + chat_history for V1, messages for V2)
        - No tools or schema instructions are added
        - Allows for unstructured responses from Cohere

    When response_model is provided:
        - Converts messages from OpenAI format to Cohere format
        - Prepends extraction instructions to the chat history (V1) or messages (V2)
        - Includes the model's JSON schema in the instructions
        - The model is instructed to extract a valid object matching the schema

    Kwargs modifications:
    - All modifications from handle_cohere_modes (message format conversion)
    - Modifies: "chat_history" (V1) or "messages" (V2) to prepend extraction instruction - only when response_model provided
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
Respond with JSON only. Do not include code fences, markdown, or extra text.
"""
    # Check client version explicitly (marker already removed by handle_cohere_modes)
    # Use presence of messages vs chat_history as indicator since marker is already consumed
    if "messages" in new_kwargs:
        # V2 format: prepend to messages
        new_kwargs["messages"].insert(0, {"role": "user", "content": instruction})
    else:
        # V1 format: prepend to chat_history
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

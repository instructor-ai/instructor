"""Claude Agent SDK utilities for instructor integration.

This module provides response handling and reask utilities for the Claude Agent SDK
integration with instructor.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def handle_claude_agent_sdk(
    response_model: type[T] | None,
    new_kwargs: dict[str, Any],
) -> tuple[type[T] | None, dict[str, Any]]:
    """Handle Claude Agent SDK response model preparation.

    The Claude Agent SDK handles JSON schema validation internally through its
    output_format option. This handler prepares the kwargs to be compatible
    with the Claude Agent SDK's query function.

    Args:
        response_model: The Pydantic model class for structured output
        new_kwargs: The keyword arguments for the API call

    Returns:
        Tuple of (response_model, updated_kwargs)
    """
    # The Claude Agent SDK uses 'prompt' instead of 'messages'
    # The actual conversion happens in the client wrapper
    # We just need to pass through the response_model information
    return response_model, new_kwargs


def reask_claude_agent_sdk(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle reask/retry for Claude Agent SDK validation errors.

    When a response fails validation, this function prepares a new request
    that includes error context to help the model correct its output.

    Args:
        kwargs: The original request kwargs
        response: The response that failed validation
        exception: The validation error that occurred

    Returns:
        Updated kwargs for retry with error context
    """
    # Get the current messages
    messages = kwargs.get("messages", [])

    # Extract error details
    error_message = str(exception)
    if hasattr(exception, "errors"):
        try:
            error_details = exception.errors()
            error_message = json.dumps(error_details, indent=2)
        except Exception:
            pass

    # Get the previous response content if available
    previous_response = ""
    if response is not None:
        if hasattr(response, "content"):
            previous_response = str(response.content)
        elif hasattr(response, "model_dump"):
            previous_response = json.dumps(response.model_dump())
        else:
            previous_response = str(response)

    # Add retry context to messages
    retry_message = {
        "role": "user",
        "content": f"""Your previous response did not match the expected schema.

Previous response:
{previous_response}

Validation error:
{error_message}

Please correct your response to match the required JSON schema exactly.
Ensure all required fields are present and have the correct types."""
    }

    messages = list(messages) + [retry_message]
    kwargs["messages"] = messages

    return kwargs

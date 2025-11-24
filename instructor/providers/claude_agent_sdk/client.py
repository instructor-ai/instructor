"""Claude Agent SDK integration for Instructor.

This module provides integration with the Claude Agent SDK, enabling structured
outputs using instructor's familiar interface with the Claude Agent SDK's
agentic capabilities.

The Claude Agent SDK already guarantees JSON schema compliance, so this integration
directly leverages that capability while providing the familiar instructor interface.
"""

from __future__ import annotations

import instructor
from typing import Any, TypeVar, Type
from pydantic import BaseModel
from collections.abc import AsyncGenerator

T = TypeVar("T", bound=BaseModel)


class ClaudeAgentSDKClient:
    """Wrapper client for Claude Agent SDK that provides an instructor-compatible interface.

    This client wraps the Claude Agent SDK's query function to work with instructor's
    response model pattern. It automatically handles:
    - Converting Pydantic models to JSON schemas for output_format
    - Iterating through the async generator to extract structured outputs
    - Validating responses against the provided response model
    """

    def __init__(self, **kwargs: Any):
        """Initialize the Claude Agent SDK client wrapper.

        Args:
            **kwargs: Default keyword arguments to pass to ClaudeAgentOptions
        """
        self.default_options = kwargs


async def claude_agent_sdk_create(
    messages: list[dict[str, Any]] | None = None,
    response_model: Type[T] | None = None,
    prompt: str | None = None,
    max_retries: int = 1,
    validation_context: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    strict: bool = True,
    hooks: Any = None,
    **kwargs: Any,
) -> T:
    """Execute a Claude Agent SDK query with structured output.

    This function provides the core integration between instructor and the
    Claude Agent SDK. It handles converting messages to prompts, setting up
    the JSON schema for structured output, and validating the response.

    Args:
        messages: List of message dicts (instructor format) - will be converted to prompt
        response_model: Pydantic model class for structured output
        prompt: Direct prompt string (alternative to messages)
        max_retries: Maximum number of retry attempts (used by instructor)
        validation_context: Additional context for validation
        context: Additional context for validation (newer parameter name)
        strict: Whether to enforce strict validation
        hooks: Instructor hooks for events
        **kwargs: Additional arguments passed to ClaudeAgentOptions

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If neither prompt nor messages are provided
        ValidationError: If the response doesn't match the expected schema
    """
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    # Convert messages to prompt if not provided directly
    if prompt is None and messages:
        # Combine messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt = "\n\n".join(prompt_parts)

    if prompt is None:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # Build options for ClaudeAgentOptions
    option_kwargs = {}

    # Pass through supported options
    for key in ["model", "max_tokens", "temperature", "working_directory"]:
        if key in kwargs:
            option_kwargs[key] = kwargs[key]

    # Get the actual response model class for schema generation
    actual_model = response_model
    if response_model is not None:
        # Handle wrapped models (e.g., Partial[Model])
        if hasattr(response_model, "__wrapped__"):
            actual_model = response_model.__wrapped__

        # Set up output_format for structured output
        option_kwargs["output_format"] = {
            "type": "json_schema",
            "schema": actual_model.model_json_schema()
        }

    options = ClaudeAgentOptions(**option_kwargs)

    # Execute the query and collect the structured output
    structured_output = None
    last_exception = None

    for attempt in range(max_retries):
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    if hasattr(message, 'structured_output') and message.structured_output:
                        structured_output = message.structured_output
                        break

            if structured_output is None:
                raise ValueError("No structured output received from Claude Agent SDK")

            # Validate the response with Pydantic
            if response_model is not None:
                # Use the actual model for validation
                validated = actual_model.model_validate(
                    structured_output,
                    context=context or validation_context,
                )
                # Attach raw response for compatibility
                validated._raw_response = structured_output
                return validated
            else:
                return structured_output

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Prepare for retry with error context
                error_context = f"\n\nPrevious attempt failed with error: {str(e)}\nPlease correct the response."
                prompt = prompt + error_context
            else:
                raise

    raise last_exception


def from_claude_agent_sdk(
    client: ClaudeAgentSDKClient | None = None,
    mode: instructor.Mode = instructor.Mode.CLAUDE_AGENT_SDK,
    **kwargs: Any,
) -> instructor.AsyncInstructor:
    """Create an AsyncInstructor instance for Claude Agent SDK.

    This function creates an instructor-compatible client that uses the Claude Agent SDK
    for structured outputs. The Claude Agent SDK provides agentic capabilities with
    guaranteed JSON schema compliance.

    Args:
        client: Optional ClaudeAgentSDKClient instance. If None, a new one is created.
        mode: The instructor mode to use. Defaults to CLAUDE_AGENT_SDK.
        **kwargs: Additional keyword arguments passed to ClaudeAgentSDKClient

    Returns:
        AsyncInstructor instance configured for Claude Agent SDK

    Example:
        ```python
        import instructor
        from instructor import from_claude_agent_sdk
        from pydantic import BaseModel
        import anyio

        class User(BaseModel):
            name: str
            age: int

        async def main():
            client = from_claude_agent_sdk()

            user = await client.create(
                response_model=User,
                messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
            )
            print(user.name)  # John
            print(user.age)   # 25

        anyio.run(main)
        ```
    """
    valid_modes = {instructor.Mode.CLAUDE_AGENT_SDK}

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="ClaudeAgentSDK",
            valid_modes=[str(m) for m in valid_modes],
        )

    if client is None:
        client = ClaudeAgentSDKClient(**kwargs)

    # Return an AsyncInstructor with our custom create function
    return instructor.AsyncInstructor(
        client=client,
        create=claude_agent_sdk_create,
        provider=instructor.Provider.CLAUDE_AGENT_SDK,
        mode=mode,
    )

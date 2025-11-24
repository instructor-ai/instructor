"""Claude Agent SDK integration for Instructor.

This module provides integration with the Claude Agent SDK, enabling structured
outputs using instructor's familiar interface with the Claude Agent SDK's
agentic capabilities.

The Claude Agent SDK already guarantees JSON schema compliance, so this integration
directly leverages that capability while providing the familiar instructor interface.

Supports both sync and async interfaces.
"""

from __future__ import annotations

import instructor
from typing import Any, TypeVar, Type, overload
from pydantic import BaseModel

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


def _convert_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert instructor message format to a single prompt string.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Combined prompt string
    """
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
    return "\n\n".join(prompt_parts)


def _prepare_options(
    response_model: Type[T] | None,
    kwargs: dict[str, Any],
) -> tuple[Type[T] | None, dict[str, Any]]:
    """Prepare ClaudeAgentOptions kwargs.

    Args:
        response_model: Pydantic model class for structured output
        kwargs: Additional kwargs to process

    Returns:
        Tuple of (actual_model, option_kwargs)
    """
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

    return actual_model, option_kwargs


async def _execute_query_async(
    prompt: str,
    option_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute the Claude Agent SDK query asynchronously.

    Args:
        prompt: The prompt to send
        option_kwargs: Options for ClaudeAgentOptions

    Returns:
        Structured output dict or None
    """
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    options = ClaudeAgentOptions(**option_kwargs)
    structured_output = None

    # Iterate through all messages to ensure proper cleanup
    # The ResultMessage with structured_output comes at the end
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if hasattr(message, 'structured_output') and message.structured_output:
                structured_output = message.structured_output
                # Don't break - let the generator complete naturally

    return structured_output


def _execute_query_sync(
    prompt: str,
    option_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute the Claude Agent SDK query synchronously.

    Uses anyio to run the async query in a sync context.

    Args:
        prompt: The prompt to send
        option_kwargs: Options for ClaudeAgentOptions

    Returns:
        Structured output dict or None
    """
    import anyio

    async def _run():
        return await _execute_query_async(prompt, option_kwargs)

    return anyio.from_thread.run_sync(_run) if anyio.get_current_task() else anyio.run(_run)


async def claude_agent_sdk_create_async(
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
    """Execute a Claude Agent SDK query with structured output (async version).

    Args:
        messages: List of message dicts (instructor format) - will be converted to prompt
        response_model: Pydantic model class for structured output
        prompt: Direct prompt string (alternative to messages)
        max_retries: Maximum number of retry attempts
        validation_context: Additional context for validation (deprecated)
        context: Additional context for validation
        strict: Whether to enforce strict validation
        hooks: Instructor hooks for events
        **kwargs: Additional arguments passed to ClaudeAgentOptions

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If neither prompt nor messages are provided
        ValidationError: If the response doesn't match the expected schema
    """
    # Convert messages to prompt if not provided directly
    if prompt is None and messages:
        prompt = _convert_messages_to_prompt(messages)

    if prompt is None:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    actual_model, option_kwargs = _prepare_options(response_model, kwargs)

    # Execute with retries
    last_exception = None
    current_prompt = prompt

    for attempt in range(max_retries):
        try:
            structured_output = await _execute_query_async(current_prompt, option_kwargs)

            if structured_output is None:
                raise ValueError("No structured output received from Claude Agent SDK")

            # Validate the response with Pydantic
            if actual_model is not None:
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
                current_prompt = f"{current_prompt}\n\nPrevious attempt failed with error: {str(e)}\nPlease correct the response."
            else:
                raise

    raise last_exception


def claude_agent_sdk_create_sync(
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
    """Execute a Claude Agent SDK query with structured output (sync version).

    This is a synchronous wrapper around the async implementation.
    Uses anyio.run() internally to execute the async code.

    Args:
        messages: List of message dicts (instructor format) - will be converted to prompt
        response_model: Pydantic model class for structured output
        prompt: Direct prompt string (alternative to messages)
        max_retries: Maximum number of retry attempts
        validation_context: Additional context for validation (deprecated)
        context: Additional context for validation
        strict: Whether to enforce strict validation
        hooks: Instructor hooks for events
        **kwargs: Additional arguments passed to ClaudeAgentOptions

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If neither prompt nor messages are provided
        ValidationError: If the response doesn't match the expected schema
    """
    import anyio

    return anyio.run(
        claude_agent_sdk_create_async,
        messages,
        response_model,
        prompt,
        max_retries,
        validation_context,
        context,
        strict,
        hooks,
        **kwargs,
    )


# Keep the original name for backwards compatibility
claude_agent_sdk_create = claude_agent_sdk_create_async


@overload
def from_claude_agent_sdk(
    client: ClaudeAgentSDKClient | None = None,
    mode: instructor.Mode = instructor.Mode.CLAUDE_AGENT_SDK,
    use_async: bool = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_claude_agent_sdk(
    client: ClaudeAgentSDKClient | None = None,
    mode: instructor.Mode = instructor.Mode.CLAUDE_AGENT_SDK,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_claude_agent_sdk(
    client: ClaudeAgentSDKClient | None = None,
    mode: instructor.Mode = instructor.Mode.CLAUDE_AGENT_SDK,
    use_async: bool = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor | instructor.Instructor:
    """Create an Instructor instance for Claude Agent SDK.

    This function creates an instructor-compatible client that uses the Claude Agent SDK
    for structured outputs. The Claude Agent SDK provides agentic capabilities with
    guaranteed JSON schema compliance.

    Args:
        client: Optional ClaudeAgentSDKClient instance. If None, a new one is created.
        mode: The instructor mode to use. Defaults to CLAUDE_AGENT_SDK.
        use_async: If True (default), returns AsyncInstructor. If False, returns sync Instructor.
        **kwargs: Additional keyword arguments passed to ClaudeAgentSDKClient

    Returns:
        AsyncInstructor or Instructor instance configured for Claude Agent SDK

    Example (async):
        ```python
        from instructor import from_claude_agent_sdk
        from pydantic import BaseModel
        import anyio

        class User(BaseModel):
            name: str
            age: int

        async def main():
            client = from_claude_agent_sdk()  # async by default

            user = await client.create(
                response_model=User,
                messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
            )
            print(user.name)  # John
            print(user.age)   # 25

        anyio.run(main)
        ```

    Example (sync):
        ```python
        from instructor import from_claude_agent_sdk
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        client = from_claude_agent_sdk(use_async=False)  # sync mode

        user = client.create(
            response_model=User,
            messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
        )
        print(user.name)  # John
        print(user.age)   # 25
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

    if use_async:
        return instructor.AsyncInstructor(
            client=client,
            create=claude_agent_sdk_create_async,
            provider=instructor.Provider.CLAUDE_AGENT_SDK,
            mode=mode,
        )
    else:
        return instructor.Instructor(
            client=client,
            create=claude_agent_sdk_create_sync,
            provider=instructor.Provider.CLAUDE_AGENT_SDK,
            mode=mode,
        )

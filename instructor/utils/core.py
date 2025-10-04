"""Core utilities for instructor library.

This module contains generic utility functions that are not provider-specific.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import AsyncGenerator, Generator, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
    get_args,
    get_origin,
)

from openai.types import CompletionUsage as OpenAIUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, create_model

# Avoid circular import - these will be imported where needed

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage

logger = logging.getLogger("instructor")
R_co = TypeVar("R_co", covariant=True)
T_Model = TypeVar("T_Model", bound=BaseModel)
T = TypeVar("T")


def extract_json_from_codeblock(content: str) -> str:
    """
    Extract JSON from a string that may contain extra text.

    The function looks for the first '{' and the last '}' in the string and
    returns the content between them, inclusive. If no braces are found,
    the original string is returned.

    Args:
        content: The string that may contain JSON

    Returns:
        The extracted JSON string
    """

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1:
        json_content = content[first_brace : last_brace + 1]
    else:
        json_content = content  # Return as is if no JSON-like content found

    return json_content


def extract_json_from_stream(
    chunks: Iterable[str],
) -> Generator[str, None, None]:
    """
    Extract JSON from a stream of chunks, handling JSON in code blocks.

    This optimized version extracts JSON from markdown code blocks or plain JSON
    by implementing a state machine approach.

    The state machine tracks several states:
    - Whether we're inside a code block (```json ... ```)
    - Whether we've started tracking a JSON object
    - Whether we're inside a string literal
    - The stack of open braces to properly identify the JSON structure

    Args:
        chunks: An iterable of string chunks

    Yields:
        Characters within the JSON object
    """
    # State flags
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    brace_stack = []
    buffer = []

    # Track potential codeblock start/end
    codeblock_buffer = []

    for chunk in chunks:
        for char in chunk:
            # Track codeblock delimiters (```)
            if not in_codeblock and char == "`":
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            elif len(codeblock_buffer) > 0 and char != "`":
                # Reset if we see something other than backticks
                codeblock_buffer = []

            # If we're in a codeblock but haven't started JSON yet
            if in_codeblock and not json_started:
                # Track end of codeblock
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                elif codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = (
                        0  # Reset if we see something other than backticks
                    )

                # Look for the start of JSON
                if char == "{":
                    json_started = True
                    brace_stack.append("{")
                    buffer.append(char)
                # Skip other characters until we find the start of JSON
                continue

            # If we've started tracking JSON
            if json_started:
                # Handle string literals and escaped characters
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                else:
                    escape_next = False

                # Track end of codeblock if we're in one
                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            # End of codeblock means end of JSON
                            in_codeblock = False
                            # Yield the buffer without the closing backticks
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break
                        continue
                    elif codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                # Track braces when not in a string
                if not in_string:
                    if char == "{":
                        brace_stack.append("{")
                    elif char == "}" and brace_stack:
                        brace_stack.pop()
                        # If we've completed a JSON object, yield its characters
                        if not brace_stack:
                            buffer.append(char)
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break

                # Add character to buffer
                buffer.append(char)
                continue

            # If we're not in a codeblock and haven't started JSON, look for standalone JSON
            if not in_codeblock and not json_started and char == "{":
                json_started = True
                brace_stack.append("{")
                buffer.append(char)

    # Yield any remaining buffer content if we have valid JSON
    if json_started and buffer:
        for c in buffer:
            yield c


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Extract JSON from an async stream of chunks, handling JSON in code blocks.

    This optimized version extracts JSON from markdown code blocks or plain JSON
    by implementing a state machine approach.

    The state machine tracks several states:
    - Whether we're inside a code block (```json ... ```)
    - Whether we've started tracking a JSON object
    - Whether we're inside a string literal
    - The stack of open braces to properly identify the JSON structure

    Args:
        chunks: An async generator yielding string chunks

    Yields:
        Characters within the JSON object
    """
    # State flags
    in_codeblock = False
    codeblock_delimiter_count = 0
    json_started = False
    in_string = False
    escape_next = False
    brace_stack = []
    buffer = []

    # Track potential codeblock start/end
    codeblock_buffer = []

    async for chunk in chunks:
        for char in chunk:
            # Track codeblock delimiters (```)
            if not in_codeblock and char == "`":
                codeblock_buffer.append(char)
                if len(codeblock_buffer) == 3:
                    in_codeblock = True
                    codeblock_delimiter_count = 0
                    codeblock_buffer = []
                continue
            elif len(codeblock_buffer) > 0 and char != "`":
                # Reset if we see something other than backticks
                codeblock_buffer = []

            # If we're in a codeblock but haven't started JSON yet
            if in_codeblock and not json_started:
                # Track end of codeblock
                if char == "`":
                    codeblock_delimiter_count += 1
                    if codeblock_delimiter_count == 3:
                        in_codeblock = False
                        codeblock_delimiter_count = 0
                    continue
                elif codeblock_delimiter_count > 0:
                    codeblock_delimiter_count = (
                        0  # Reset if we see something other than backticks
                    )

                # Look for the start of JSON
                if char == "{":
                    json_started = True
                    brace_stack.append("{")
                    buffer.append(char)
                # Skip other characters until we find the start of JSON
                continue

            # If we've started tracking JSON
            if json_started:
                # Handle string literals and escaped characters
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif char == "\\" and in_string:
                    escape_next = True
                    buffer.append(char)
                    continue
                else:
                    escape_next = False

                # Track end of codeblock if we're in one
                if in_codeblock and not in_string:
                    if char == "`":
                        codeblock_delimiter_count += 1
                        if codeblock_delimiter_count == 3:
                            # End of codeblock means end of JSON
                            in_codeblock = False
                            # Yield the buffer without the closing backticks
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break
                        continue
                    elif codeblock_delimiter_count > 0:
                        codeblock_delimiter_count = 0

                # Track braces when not in a string
                if not in_string:
                    if char == "{":
                        brace_stack.append("{")
                    elif char == "}" and brace_stack:
                        brace_stack.pop()
                        # If we've completed a JSON object, yield its characters
                        if not brace_stack:
                            buffer.append(char)
                            for c in buffer:
                                yield c
                            buffer = []
                            json_started = False
                            break

                # Add character to buffer
                buffer.append(char)
                continue

            # If we're not in a codeblock and haven't started JSON, look for standalone JSON
            if not in_codeblock and not json_started and char == "{":
                json_started = True
                brace_stack.append("{")
                buffer.append(char)

    # Yield any remaining buffer content if we have valid JSON
    if json_started and buffer:
        for c in buffer:
            yield c


def update_total_usage(
    response: T_Model | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Model | ChatCompletion | None:
    if response is None:
        return None

    response_usage = getattr(response, "usage", None)
    if isinstance(response_usage, OpenAIUsage) and isinstance(total_usage, OpenAIUsage):
        total_usage.completion_tokens += response_usage.completion_tokens or 0
        total_usage.prompt_tokens += response_usage.prompt_tokens or 0
        total_usage.total_tokens += response_usage.total_tokens or 0
        if (rtd := response_usage.completion_tokens_details) and (
            ttd := total_usage.completion_tokens_details
        ):
            ttd.audio_tokens = (ttd.audio_tokens or 0) + (rtd.audio_tokens or 0)
            ttd.reasoning_tokens = (ttd.reasoning_tokens or 0) + (
                rtd.reasoning_tokens or 0
            )
        if (rpd := response_usage.prompt_tokens_details) and (
            tpd := total_usage.prompt_tokens_details
        ):
            tpd.audio_tokens = (tpd.audio_tokens or 0) + (rpd.audio_tokens or 0)
            tpd.cached_tokens = (tpd.cached_tokens or 0) + (rpd.cached_tokens or 0)
        response.usage = total_usage  # Replace each response usage with the total usage
        return response

    # Anthropic usage.
    try:
        from anthropic.types import Usage as AnthropicUsage

        if isinstance(response_usage, AnthropicUsage) and isinstance(
            total_usage, AnthropicUsage
        ):
            if not total_usage.cache_creation_input_tokens:
                total_usage.cache_creation_input_tokens = 0

            if not total_usage.cache_read_input_tokens:
                total_usage.cache_read_input_tokens = 0

            total_usage.input_tokens += response_usage.input_tokens or 0
            total_usage.output_tokens += response_usage.output_tokens or 0
            total_usage.cache_creation_input_tokens += (
                response_usage.cache_creation_input_tokens or 0
            )
            total_usage.cache_read_input_tokens += (
                response_usage.cache_read_input_tokens or 0
            )
            response.usage = total_usage
            return response
    except ImportError:
        pass

    logger.debug("No compatible response.usage found, token usage not updated.")
    return response


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if (
        hasattr(message, "function_call")
        and message.function_call is not None
        and ret["content"]
    ):
        if not isinstance(ret["content"], str):
            response_message: str = ""
            for content_message in ret["content"]:
                if isinstance(content_message, dict):
                    # Use get() to safely access values
                    message_type = content_message.get("type")
                    if message_type == "text":
                        text_content = content_message.get("text", "")
                        response_message += text_content
                    elif message_type == "refusal":
                        refusal_content = content_message.get("refusal", "")
                        response_message += refusal_content
            ret["content"] = response_message
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def is_async(func: Callable[..., Any]) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__  # type: ignore - dynamic
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine


def merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge consecutive messages from the same role into a single message.

    This optimized version pre-allocates the result list and minimizes operations.

    Args:
        messages: List of message dictionaries to merge

    Returns:
        List of merged message dictionaries
    """
    if not messages:
        return []

    # Pre-allocate result list with estimated size (worst case: no merges happen)
    message_count = len(messages)
    new_messages = []

    # Detect whether all messages have a flat content (i.e. all string)
    # Some providers require content to be a string, so we need to check that and behave accordingly
    # Fast path: avoid checking all messages if the first few have mixed content types
    flat_string = True
    for _i, m in enumerate(messages[: min(10, message_count)]):
        if not isinstance(m.get("content", ""), str):
            flat_string = False
            break

    # Only check all messages if we haven't determined it's not flat_string
    if flat_string and message_count > 10:
        flat_string = all(isinstance(m.get("content", ""), str) for m in messages[10:])

    # Process messages with a single loop
    for message in messages:
        role = message.get("role", "user")
        new_content = message.get("content", "")

        # Transform string content to list if needed
        if not flat_string and isinstance(new_content, str):
            new_content = [{"type": "text", "text": new_content}]

        # Check if we can merge with previous message
        if new_messages and role == new_messages[-1]["role"]:
            if flat_string:
                # Fast path for string content
                new_messages[-1]["content"] += f"\n\n{new_content}"
            else:
                # Fast path for list content
                if isinstance(new_content, list):
                    new_messages[-1]["content"].extend(new_content)
                else:
                    # Fallback for unexpected content type
                    new_messages[-1]["content"].append(new_content)
        else:
            # Add new message
            new_messages.append({"role": role, "content": new_content})

    return new_messages


class classproperty(Generic[R_co]):
    """Descriptor for class-level properties.

    Examples:
        >>> from instructor.utils import classproperty

        >>> class MyClass:
        ...     @classproperty
        ...     def my_property(cls):
        ...         return cls

        >>> assert MyClass.my_property
    """

    def __init__(self, method: Callable[[Any], R_co]) -> None:
        self.cproperty = method

    def __get__(self, instance: object, cls: type[Any]) -> R_co:
        return self.cproperty(cls)


def get_message_content(message: ChatCompletionMessageParam) -> list[Any]:
    """
    Extract content from a message and ensure it's returned as a list.

    This optimized version handles different message formats more efficiently.

    Args:
        message: A message in ChatCompletionMessageParam format

    Returns:
        The message content as a list
    """
    # Fast path for empty message
    if not message:
        return [""]

    # Get content with default empty string
    content = message.get("content", "")

    # Fast path for common content types
    if isinstance(content, list):
        return content if content else [""]

    # Return single item list with content (could be string, None, or other)
    return [content if content is not None else ""]


def disable_pydantic_error_url():
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def is_simple_type(typehint: type[T]) -> bool:
    """Check if a type is a simple type that can be adapted."""
    from instructor.dsl.simple_type import is_simple_type as _is_simple_type

    return _is_simple_type(typehint)


def prepare_response_model(response_model: type[T] | None) -> type[T] | None:
    """
    Prepares the response model for use in the API call.

    This function performs several transformations on the input response_model:
    1. If the response_model is None, it returns None.
    2. If it's a simple type, it wraps it in a ModelAdapter.
    3. If it's a TypedDict, it converts it to a Pydantic BaseModel.
    4. If it's an Iterable, it wraps the element type in an IterableModel.
    5. If it's not already a subclass of OpenAISchema, it applies the openai_schema decorator.

    Args:
        response_model (type[T] | None): The input response model to be prepared.

    Returns:
        type[T] | None: The prepared response model, or None if the input was None.
    """
    if response_model is None:
        return None

    if is_simple_type(response_model):
        from instructor.dsl.simple_type import ModelAdapter

        response_model = ModelAdapter[response_model]

    if is_typed_dict(response_model):
        response_model: BaseModel = create_model(
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},
        )

    if get_origin(response_model) is Iterable:
        from instructor.dsl.iterable import IterableModel

        iterable_element_class = get_args(response_model)[0]
        response_model = IterableModel(iterable_element_class)

    # Import here to avoid circular dependency
    from ..processing.function_calls import OpenAISchema, openai_schema

    if not issubclass(response_model, OpenAISchema):
        response_model = openai_schema(response_model)  # type: ignore

    return response_model

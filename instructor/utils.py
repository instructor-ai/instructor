from __future__ import annotations

import inspect
import json
import logging
import os
import re
from collections.abc import AsyncGenerator, Generator, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

from instructor.multimodal import PDF, Image, Audio
from openai.types import CompletionUsage as OpenAIUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage

from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger("instructor")
R_co = TypeVar("R_co", covariant=True)
T_Model = TypeVar("T_Model", bound="Response")


class Response(Protocol):
    usage: OpenAIUsage | AnthropicUsage


class Provider(Enum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    GENAI = "genai"
    DATABRICKS = "databricks"
    CEREBRAS = "cerebras"
    FIREWORKS = "fireworks"
    WRITER = "writer"
    UNKNOWN = "unknown"
    BEDROCK = "bedrock"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"


def get_provider(base_url: str) -> Provider:
    if "anyscale" in str(base_url):
        return Provider.ANYSCALE
    elif "together" in str(base_url):
        return Provider.TOGETHER
    elif "anthropic" in str(base_url):
        return Provider.ANTHROPIC
    elif "cerebras" in str(base_url):
        return Provider.CEREBRAS
    elif "fireworks" in str(base_url):
        return Provider.FIREWORKS
    elif "groq" in str(base_url):
        return Provider.GROQ
    elif "openai" in str(base_url):
        return Provider.OPENAI
    elif "mistral" in str(base_url):
        return Provider.MISTRAL
    elif "cohere" in str(base_url):
        return Provider.COHERE
    elif "gemini" in str(base_url):
        return Provider.GEMINI
    elif "databricks" in str(base_url):
        return Provider.DATABRICKS
    elif "vertexai" in str(base_url):
        return Provider.VERTEXAI
    elif "writer" in str(base_url):
        return Provider.WRITER
    elif "perplexity" in str(base_url):
        return Provider.PERPLEXITY
    elif "openrouter" in str(base_url):
        return Provider.OPENROUTER
    return Provider.UNKNOWN


# Regex patterns for JSON extraction
_JSON_CODEBLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_PATTERN = re.compile(r"({[\s\S]*})")


def extract_json_from_codeblock(content: str) -> str:
    """
    Extract JSON from a string that may contain markdown code blocks or plain JSON.

    This optimized version uses regex patterns to extract JSON more efficiently.

    Args:
        content: The string that may contain JSON

    Returns:
        The extracted JSON string
    """
    # First try to find JSON in code blocks
    match = _JSON_CODEBLOCK_PATTERN.search(content)
    if match:
        json_content = match.group(1).strip()
    else:
        # Look for JSON objects with the pattern { ... }
        match = _JSON_PATTERN.search(content)
        if match:
            json_content = match.group(1)
        else:
            # Fallback to the old method if regex doesn't find anything
            first_paren = content.find("{")
            last_paren = content.rfind("}")
            if first_paren != -1 and last_paren != -1:
                json_content = content[first_paren : last_paren + 1]
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
                if "text" in content_message:
                    response_message += content_message["text"]
                elif "refusal" in content_message:
                    response_message += content_message["refusal"]
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


def transform_to_gemini_prompt(
    messages_chatgpt: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    """
    Transform messages from OpenAI format to Gemini format.

    This optimized version reduces redundant processing and improves
    handling of system messages.

    Args:
        messages_chatgpt: Messages in OpenAI format

    Returns:
        Messages in Gemini format
    """
    # Fast path for empty messages
    if not messages_chatgpt:
        return []

    # Process system messages first (collect all system messages)
    system_prompts = []
    for message in messages_chatgpt:
        if message.get("role") == "system":
            content = message.get("content", "")
            if content:  # Only add non-empty system prompts
                system_prompts.append(content)

    # Format system prompt if we have any
    system_prompt = ""
    if system_prompts:
        # Handle multiple system prompts by joining them
        system_prompt = "\n\n".join(filter(None, system_prompts))

    # Count non-system messages to pre-allocate result list
    message_count = sum(1 for m in messages_chatgpt if m.get("role") != "system")
    messages_gemini = []

    # Role mapping for faster lookups
    role_map = {
        "user": "user",
        "assistant": "model",
    }

    # Process non-system messages in one pass
    for message in messages_chatgpt:
        role = message.get("role", "")
        if role in role_map:
            gemini_role = role_map[role]
            messages_gemini.append(
                {"role": gemini_role, "parts": get_message_content(message)}
            )

    # Add system prompt if we have one
    if system_prompt:
        if messages_gemini:
            # Add to the first message (most likely user message)
            first_message = messages_gemini[0]
            # Only insert if parts is a list
            if isinstance(first_message.get("parts"), list):
                first_message["parts"].insert(0, f"*{system_prompt}*")
        else:
            # Create a new user message just for the system prompt
            messages_gemini.append({"role": "user", "parts": [f"*{system_prompt}*"]})

    return messages_gemini


def map_to_gemini_function_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Map OpenAPI schema to Gemini properties: gemini function call schemas

    Ref - https://ai.google.dev/api/python/google/generativeai/protos/Schema,
    Note that `enum` requires specific `format` setting
    """

    import jsonref

    class FunctionSchema(BaseModel):
        description: str | None = None
        enum: list[str] | None = None
        example: Any | None = None
        format: str | None = None
        nullable: bool | None = None
        items: FunctionSchema | None = None
        required: list[str] | None = None
        type: str
        properties: dict[str, FunctionSchema] | None = None

    schema: dict[str, Any] = jsonref.replace_refs(obj, lazy_load=False)  # type: ignore
    schema.pop("$defs", "")

    def add_enum_format(obj: dict[str, Any]) -> dict[str, Any]:
        if isinstance(obj, dict):
            new_dict: dict[str, Any] = {}
            for key, value in obj.items():
                new_dict[key] = add_enum_format(value)
                if key == "enum":
                    new_dict["format"] = "enum"
            return new_dict
        else:
            return obj

    schema = add_enum_format(schema)

    return FunctionSchema(**schema).model_dump(exclude_none=True, exclude_unset=True)


def update_gemini_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Update keyword arguments for Gemini API from OpenAI format.

    This optimized version reduces redundant operations and uses
    efficient data transformations.

    Args:
        kwargs: Dictionary of keyword arguments to update

    Returns:
        Updated dictionary of keyword arguments
    """
    # Make a copy of kwargs to avoid modifying the original
    result = kwargs.copy()

    # Mapping of OpenAI args to Gemini args - defined as constant
    # for quicker lookup without recreating the dictionary on each call
    OPENAI_TO_GEMINI_MAP = {
        "max_tokens": "max_output_tokens",
        "temperature": "temperature",
        "n": "candidate_count",
        "top_p": "top_p",
        "stop": "stop_sequences",
    }

    # Update generation_config if present
    if "generation_config" in result:
        gen_config = result["generation_config"]

        # Bulk process the mapping with fewer conditionals
        for openai_key, gemini_key in OPENAI_TO_GEMINI_MAP.items():
            if openai_key in gen_config:
                val = gen_config.pop(openai_key)
                if val is not None:  # Only set if value is not None
                    gen_config[gemini_key] = val

    # Transform messages format if messages key exists
    if "messages" in result:
        # Transform messages and store them under "contents" key
        result["contents"] = transform_to_gemini_prompt(result.pop("messages"))

    # Handle safety settings - import here to avoid circular imports
    from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore

    # Create or get existing safety settings
    safety_settings = result.get("safety_settings", {})
    result["safety_settings"] = safety_settings

    # Define default safety thresholds - these are static and can be
    # defined once rather than recreating the dict on each call
    DEFAULT_SAFETY_THRESHOLDS = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # Update safety settings with defaults if needed (more efficient loop)
    for category, threshold in DEFAULT_SAFETY_THRESHOLDS.items():
        current = safety_settings.get(category)
        # Only update if not set or less restrictive than default
        # Note: Lower values are more restrictive in HarmBlockThreshold
        # BLOCK_NONE = 0, BLOCK_LOW_AND_ABOVE = 1, BLOCK_MEDIUM_AND_ABOVE = 2, BLOCK_ONLY_HIGH = 3
        if current is None or current > threshold:
            safety_settings[category] = threshold

    return result


def disable_pydantic_error_url():
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"


class SystemMessage(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict[str, str]


def combine_system_messages(
    existing_system: Union[str, list[SystemMessage], None],  # noqa: UP007
    new_system: Union[str, list[SystemMessage]],  # noqa: UP007
) -> Union[str, list[SystemMessage]]:  # noqa: UP007
    """
    Combine existing and new system messages.

    This optimized version uses a more direct approach with fewer branches.

    Args:
        existing_system: Existing system message(s) or None
        new_system: New system message(s) to add

    Returns:
        Combined system message(s)
    """
    # Fast path for None existing_system (avoid unnecessary operations)
    if existing_system is None:
        return new_system

    # Validate input types
    if not isinstance(existing_system, (str, list)) or not isinstance(
        new_system, (str, list)
    ):
        raise ValueError(
            f"System messages must be strings or lists, got {type(existing_system)} and {type(new_system)}"
        )

    # Use direct type comparison instead of isinstance for better performance
    if isinstance(existing_system, str) and isinstance(new_system, str):
        # Both are strings, join with newlines
        # Avoid creating intermediate strings by joining only once
        return f"{existing_system}\n\n{new_system}"
    elif isinstance(existing_system, list) and isinstance(new_system, list):
        # Both are lists, use list extension in place to avoid creating intermediate lists
        # First create a new list to avoid modifying the original
        result = list(existing_system)
        result.extend(new_system)
        return result
    elif isinstance(existing_system, str) and isinstance(new_system, list):
        # existing is string, new is list
        # Create a pre-sized list to avoid resizing
        result = [SystemMessage(type="text", text=existing_system)]
        result.extend(new_system)
        return result
    elif isinstance(existing_system, list) and isinstance(new_system, str):
        # existing is list, new is string
        # Create message once and add to existing
        new_message = SystemMessage(type="text", text=new_system)
        result = list(existing_system)
        result.append(new_message)
        return result

    # This should never happen due to validation above
    return existing_system


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    """
    Extract system messages from a list of messages.

    This optimized version pre-allocates the result list and
    reduces function call overhead.

    Args:
        messages: List of messages to extract system messages from

    Returns:
        List of system messages
    """
    # Fast path for empty messages
    if not messages:
        return []

    # First count system messages to pre-allocate result list
    system_count = sum(1 for m in messages if m.get("role") == "system")

    # If no system messages, return empty list
    if system_count == 0:
        return []

    # Helper function to convert a message content to SystemMessage
    def convert_message(content: Any) -> SystemMessage:
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        elif isinstance(content, dict):
            return SystemMessage(**content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    # Process system messages
    result: list[SystemMessage] = []

    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")

            # Skip empty content
            if not content:
                continue

            # Handle list or single content
            if isinstance(content, list):
                # Process each item in the list
                for item in content:
                    if item:  # Skip empty items
                        result.append(convert_message(item))
            else:
                # Process single content
                result.append(convert_message(content))

    return result


def extract_genai_system_message(
    messages: list[dict[str, Any]],
) -> str:
    """
    Extract system messages from a list of messages.

    We expect an explicit system messsage for this provider.
    """
    system_messages = ""

    for message in messages:
        if isinstance(message, str):
            continue
        elif isinstance(message, dict):
            if message.get("role") == "system":
                if isinstance(message.get("content"), str):
                    system_messages += message.get("content", "") + "\n\n"
                elif isinstance(message.get("content"), list):
                    for item in message.get("content", []):
                        if isinstance(item, str):
                            system_messages += item + "\n\n"

    return system_messages


def convert_to_genai_messages(
    messages: list[Union[str, dict[str, Any], list[dict[str, Any]]]],  # noqa: UP007
) -> list[Any]:
    """
    Convert a list of messages to a list of dictionaries in the format expected by the Gemini API.

    This optimized version pre-allocates the result list and
    reduces function call overhead.
    """
    from google.genai import types

    result: list[Union[types.Content, types.File]] = []  # noqa: UP007

    for message in messages:
        # We assume this is the user's message and we don't need to convert it
        if isinstance(message, str):
            result.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            )
        elif isinstance(message, types.Content):
            result.append(message)
        elif isinstance(message, types.File):
            result.append(message)
        elif isinstance(message, dict):
            assert "role" in message
            assert "content" in message

            if message["role"] == "system":
                continue

            if message["role"] not in {"user", "model"}:
                raise ValueError(f"Unsupported role: {message['role']}")

            if isinstance(message["content"], str):
                result.append(
                    types.Content(
                        role=message["role"],
                        parts=[types.Part.from_text(text=message["content"])],
                    )
                )

            elif isinstance(message["content"], list):
                content_parts = []

                for content_item in message["content"]:
                    if isinstance(content_item, str):
                        content_parts.append(types.Part.from_text(text=content_item))
                    elif isinstance(content_item, (Image, Audio, PDF)):
                        content_parts.append(content_item.to_genai())
                    else:
                        raise ValueError(
                            f"Unsupported content item type: {type(content_item)}"
                        )

                result.append(
                    types.Content(
                        role=message["role"],
                        parts=content_parts,
                    )
                )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return result

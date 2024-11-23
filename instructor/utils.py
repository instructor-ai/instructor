from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import AsyncGenerator, Generator, Iterable
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    cast,
    Dict,
)

# Initialize safety settings with default values
harm_settings: Dict[str, str] = {}
block_high: str | None = None

# Try to import and set Gemini-specific values
try:
    from vertexai.generative_models import generative_models  # type: ignore
    harm_settings = {
        "hate_speech": "HARM_CATEGORY_HATE_SPEECH",
        "harassment": "HARM_CATEGORY_HARASSMENT",
        "dangerous": "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    block_high = "BLOCK_ONLY_HIGH"
except ImportError:
    pass  # Use default values

from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam

logger = logging.getLogger("instructor")

R_co = TypeVar("R_co", covariant=True)

class TokenDetails(TypedDict, total=False):
    """Token details for completion tokens."""
    audio_tokens: int
    reasoning_tokens: int
    cached_tokens: int

class CompletionTokensDetails(TypedDict, total=False):
    """Details about completion tokens."""
    completion_tokens_details: dict[str, int]

class OpenAIUsageDict(TypedDict, total=False):
    """OpenAI usage information."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: dict[str, int]

class AnthropicUsageDict(TypedDict, total=False):
    """Anthropic usage information."""
    input_tokens: int
    output_tokens: int
    completion_tokens_details: dict[str, int]

# Define the union type for all usage dictionaries
UsageDict = Union[OpenAIUsageDict, AnthropicUsageDict, CompletionTokensDetails]

T_Model = TypeVar("T_Model", bound="ResponseProtocol")

class ResponseProtocol(Protocol):
    """Protocol for response objects."""
    usage: UsageDict

class Response(Generic[R_co]):
    """Response object that wraps the response from the API."""
    def __init__(self, response: R_co) -> None:
        self._response: R_co = response
        self._usage: UsageDict | None = None

    @property
    def usage(self) -> UsageDict:
        """Get the usage information from the response."""
        if self._usage is None:
            self._usage = cast(UsageDict, {})
            if hasattr(self._response, "usage"):
                self._usage = cast(UsageDict, getattr(self._response, "usage", {}))
        return self._usage

    @usage.setter
    def usage(self, value: UsageDict) -> None:
        """Set the usage information."""
        self._usage = value

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the response."""
        return getattr(self._response, name)

    def __repr__(self) -> str:
        """Get the string representation of the response."""
        return repr(self._response)


class Provider(str, Enum):
    """Enum for different LLM providers."""
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    ANTHROPIC = "anthropic"
    CEREBRAS = "cerebras"
    FIREWORKS = "fireworks"
    GROQ = "groq"
    OPENAI = "openai"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    DATABRICKS = "databricks"
    VERTEXAI = "vertexai"
    WRITER = "writer"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


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
    return Provider.UNKNOWN


def extract_json_from_codeblock(content: str) -> str:
    first_paren = content.find("{")
    last_paren = content.rfind("}")
    return content[first_paren : last_paren + 1]


def extract_json_from_stream(chunks: Iterable[str]) -> Generator[str, None, None]:
    capturing = False
    brace_count = 0
    for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    capturing = False
    brace_count = 0
    async for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


def update_total_usage(
    response: Response[Any] | None,
    total_usage: UsageDict,
) -> Response[Any] | None:
    """Update total token usage with a new response's usage information."""
    if response is None:
        return None

    response_usage: Dict[str, Any] = getattr(response, "usage", {})
    if not isinstance(response_usage, dict) or not isinstance(total_usage, dict):
        logger.debug("No compatible response.usage found, token usage not updated.")
        return response

    # Handle OpenAI usage
    openai_keys = ["completion_tokens", "prompt_tokens", "total_tokens"]
    for key in openai_keys:
        if key in response_usage:
            total_usage[key] = int(total_usage.get(key, 0)) + int(response_usage[key])

    # Handle token details
    token_details: Dict[str, int] = {}
    if "completion_tokens_details" in response_usage:
        raw_details: Dict[str, Any] = dict(response_usage["completion_tokens_details"])
        for k, v in raw_details.items():
            if isinstance(v, (int, float)):
                token_details[str(k)] = int(v)

        if token_details:
            if "completion_tokens_details" not in total_usage:
                total_usage["completion_tokens_details"] = {}

            details = total_usage["completion_tokens_details"]
            for key in ["audio_tokens", "reasoning_tokens", "cached_tokens"]:
                if key in token_details:
                    details[key] = details.get(key, 0) + token_details[key]

    # Handle Anthropic usage
    anthropic_keys = ["input_tokens", "output_tokens"]
    if all(key in response_usage for key in anthropic_keys):
        for key in anthropic_keys:
            if key in response_usage:
                val = int(response_usage[key]) if response_usage[key] is not None else 0
                total_usage[key] = int(total_usage.get(key, 0)) + val

    setattr(response, "usage", total_usage)
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
    # merge all consecutive user messages into a single message
    new_messages: list[dict[str, Any]] = []
    # Detect whether all messages have a flat content (i.e. all string)
    # Some providers require content to be a string, so we need to check that and behave accordingly
    flat_string = all(isinstance(m["content"], str) for m in messages)
    for message in messages:
        new_content = message["content"]
        if not flat_string and isinstance(new_content, str):
            # If content is not flat, transform it into a list of text
            new_content = [{"type": "text", "text": new_content}]

        if len(new_messages) > 0 and message["role"] == new_messages[-1]["role"]:
            if flat_string:
                # New content is a string
                new_messages[-1]["content"] += f"\n\n{new_content}"
            else:
                # New content is a list
                new_messages[-1]["content"].extend(new_content)
        else:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": new_content,
                }
            )

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
    content = message.get("content", "")
    try:
        if isinstance(content, list):
            return content
        else:
            return [content]
    except Exception as e:
        logging.debug(f"Error getting message content: {e}")
        return [content]


def transform_to_gemini_prompt(
    messages_chatgpt: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    messages_gemini: list[dict[str, Any]] = []
    system_prompt = ""
    for message in messages_chatgpt:
        if message["role"] == "system":
            system_prompt = message["content"]
        elif message["role"] == "user":
            messages_gemini.append(
                {"role": "user", "parts": get_message_content(message)}
            )
        elif message["role"] == "assistant":
            messages_gemini.append(
                {"role": "model", "parts": get_message_content(message)}
            )

    if system_prompt:
        if messages_gemini:
            messages_gemini[0]["parts"].insert(0, f"*{system_prompt}*")
        else:
            messages_gemini.append({"role": "user", "parts": [f"*{system_prompt}*"]})

    return messages_gemini


def map_to_gemini_function_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Map OpenAPI schema to Gemini properties: gemini function call schemas

    Ref - https://ai.google.dev/api/python/google/generativeai/protos/Schema,
    Note that `enum` requires specific `format` setting
    """
    try:
        from typing import cast
        import jsonref
    except ImportError:
        raise ImportError("jsonref is required for function schema mapping. Please install it with 'pip install jsonref'")

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

    schema_str = json.dumps(obj)
    resolved: dict[str, Any] = cast(dict[str, Any], jsonref.loads(schema_str))
    schema = resolved
    schema.pop("$defs", None)

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
    """Update kwargs for Gemini API compatibility."""
    if "generation_config" in kwargs:
        map_openai_args_to_gemini = {
            "max_tokens": "max_output_tokens",
            "temperature": "temperature",
            "n": "candidate_count",
            "top_p": "top_p",
            "stop": "stop_sequences",
        }

        # update gemini config if any params are set
        for k, v in map_openai_args_to_gemini.items():
            val = kwargs["generation_config"].pop(k, None)
            if val is None:
                continue
            kwargs["generation_config"][v] = val

    kwargs["contents"] = transform_to_gemini_prompt(kwargs.pop("messages"))

    # minimize gemini safety related errors - model is highly prone to false alarms
    try:
        if not harm_settings:
            logger.debug("vertexai not available, skipping safety settings")
            return kwargs

        safety_settings = kwargs.get("safety_settings", {})

        # Use pre-defined constants for safety settings
        for harm_category in harm_settings.values():
            if harm_category not in safety_settings:
                safety_settings[harm_category] = block_high

        kwargs["safety_settings"] = safety_settings
    except Exception:
        logger.debug("Error configuring safety settings", exc_info=True)

    return kwargs


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
    if existing_system is None:
        return new_system

    if isinstance(existing_system, str) and isinstance(new_system, str):
        return f"{existing_system}\n\n{new_system}"

    if isinstance(existing_system, list) and isinstance(new_system, list):
        return existing_system + new_system

    if isinstance(existing_system, str) and isinstance(new_system, list):
        return [SystemMessage(type="text", text=existing_system)] + new_system

    if isinstance(existing_system, list) and isinstance(new_system, str):
        return existing_system + [SystemMessage(type="text", text=new_system)]

    raise ValueError("Unsupported system message type combination")


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    def convert_message(content: Union[str, dict[str, Any]]) -> SystemMessage:  # noqa: UP007
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        elif isinstance(content, dict):
            return SystemMessage(**content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    result: list[SystemMessage] = []
    for m in messages:
        if m["role"] == "system":
            # System message must always be a string or list of dictionaries
            content = cast(Union[str, list[dict[str, Any]]], m["content"])  # noqa: UP007
            if isinstance(content, list):
                result.extend(convert_message(item) for item in content)
            else:
                result.append(convert_message(content))
    return result

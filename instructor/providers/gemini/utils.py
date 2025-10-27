"""Google-specific utilities (Gemini, GenAI, VertexAI).

This module contains utilities specific to Google providers,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Union

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ...dsl.partial import Partial, PartialBase
from ...core.exceptions import ConfigurationError
from ...mode import Mode
from ...processing.multimodal import Audio, Image, PDF
from ...utils.core import get_message_content

if TYPE_CHECKING:
    from google.genai import types


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


def verify_no_unions(obj: dict[str, Any]) -> bool:
    """
    Verify that the object does not contain any Union types (except Optional and Decimal).
    Optional[T] is allowed as it becomes Union[T, None].
    Decimal types are allowed as Union[str, float] or Union[float, str].
    """
    for prop_value in obj["properties"].values():
        if "anyOf" in prop_value:
            any_of_list = prop_value["anyOf"]
            if not isinstance(any_of_list, list) or len(any_of_list) != 2:
                return False

            # Extract the types from the anyOf list
            types_in_union = []
            for item in any_of_list:
                if isinstance(item, dict) and "type" in item:
                    types_in_union.append(item["type"])

            # Check if this is an Optional type (Union with None/null)
            if "null" in types_in_union:
                # This is Optional[T] - allow it
                continue

            # Check if this is a Decimal type (Union of string and number)
            if set(types_in_union) == {"string", "number"}:
                # This is a Decimal type (string | number) - allow it
                continue

            # This is some other Union type - reject it
            return False

        if "properties" in prop_value and not verify_no_unions(prop_value):
            return False

    return True


def map_to_gemini_function_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Map OpenAPI schema to Gemini function call schema.

    Transforms a standard JSON schema to Gemini's expected format:
    - Adds 'format': 'enum' for enum fields
    - Converts Optional[T] (anyOf with null) to nullable fields
    - Rejects true Union types (non-Optional anyOf)

    Ref: https://ai.google.dev/api/python/google/generativeai/protos/Schema
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
        type: str | None = None
        anyOf: list[dict[str, Any]] | None = None
        properties: dict[str, FunctionSchema] | None = None

    # Resolve any $ref references in the schema
    schema: dict[str, Any] = jsonref.replace_refs(obj, lazy_load=False)  # type: ignore
    schema.pop("$defs", None)

    def transform_schema_node(node: Any) -> Any:
        """Transform a single schema node recursively."""
        if isinstance(node, list):
            return [transform_schema_node(item) for item in node]

        if not isinstance(node, dict):
            return node

        transformed = {}

        for key, value in node.items():
            if key == "enum":
                # Gemini requires 'format': 'enum' for enum fields
                transformed[key] = value
                transformed["format"] = "enum"
            elif key == "anyOf" and isinstance(value, list) and len(value) == 2:
                # Handle Optional[T] which becomes Union[T, None] in JSON schema
                non_null_items = [
                    item
                    for item in value
                    if not (isinstance(item, dict) and item.get("type") == "null")
                ]

                if len(non_null_items) == 1:
                    # This is Optional[T] - merge the actual type and mark as nullable
                    actual_type = transform_schema_node(non_null_items[0])
                    transformed.update(actual_type)
                    transformed["nullable"] = True
                else:
                    # Check if this is a Decimal type (string | number)
                    types_in_union = []
                    for item in value:
                        if isinstance(item, dict) and "type" in item:
                            types_in_union.append(item["type"])

                    if set(types_in_union) == {"string", "number"}:
                        # This is a Decimal type - keep the anyOf structure
                        transformed[key] = transform_schema_node(value)
                    else:
                        # This is a true Union type - keep as is and let validation catch it
                        transformed[key] = transform_schema_node(value)
            else:
                transformed[key] = transform_schema_node(value)

        return transformed

    schema = transform_schema_node(schema)

    # Validate that no unsupported Union types remain
    if not verify_no_unions(schema):
        raise ValueError(
            "Gemini does not support Union types (except Optional). Please change your function schema"
        )

    return FunctionSchema(**schema).model_dump(exclude_none=True, exclude_unset=True)


def update_genai_kwargs(
    kwargs: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Update keyword arguments for google.genai package from OpenAI format.
    """
    from google.genai.types import HarmBlockThreshold, HarmCategory

    new_kwargs = kwargs.copy()

    OPENAI_TO_GEMINI_MAP = {
        "max_tokens": "max_output_tokens",
        "temperature": "temperature",
        "n": "candidate_count",
        "top_p": "top_p",
        "stop": "stop_sequences",
        "seed": "seed",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
    }

    generation_config = new_kwargs.pop("generation_config", {})

    for openai_key, gemini_key in OPENAI_TO_GEMINI_MAP.items():
        if openai_key in generation_config:
            val = generation_config.pop(openai_key)
            if val is not None:  # Only set if value is not None
                base_config[gemini_key] = val

    safety_settings = new_kwargs.pop("safety_settings", {})
    base_config["safety_settings"] = []

    # Filter out image related harm categories which are not
    # supported for text based models
    supported_categories = [
        c
        for c in HarmCategory
        if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED
        and not c.name.startswith("HARM_CATEGORY_IMAGE_")
    ]

    for category in supported_categories:
        threshold = safety_settings.get(category, HarmBlockThreshold.OFF)
        base_config["safety_settings"].append(
            {
                "category": category,
                "threshold": threshold,
            }
        )

    # Handle thinking_config parameter - pass through directly since it's already in genai format
    thinking_config = new_kwargs.pop("thinking_config", None)
    if thinking_config is not None:
        base_config["thinking_config"] = thinking_config

    return base_config


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
    try:
        from google.genai.types import HarmBlockThreshold, HarmCategory  # type: ignore
    except ImportError:
        # Fallback for backward compatibility
        from google.generativeai.types import (  # type: ignore
            HarmBlockThreshold,
            HarmCategory,
        )

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

    if system_messages and len(messages) == 1:
        raise ValueError(
            "At least one user message must be included. A system message alone is not sufficient."
        )

    if re.search(r"{{.*?}}|{%.*?%}", system_messages):
        raise ValueError(
            "Jinja templating is not supported in system messages with Google GenAI, only user messages."
        )

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


# Reask functions
def reask_gemini_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Gemini
    exception: Exception,
):
    """
    Handle reask for Gemini tools mode when validation fails.

    Kwargs modifications:
    - Adds: "contents" (tool response messages indicating validation errors)
    """
    from google.ai import generativelanguage as glm  # type: ignore

    reask_msgs = [
        {
            "role": "model",
            "parts": [
                glm.FunctionCall(
                    name=response.parts[0].function_call.name,
                    args=response.parts[0].function_call.args,
                )
            ],
        },
        {
            "role": "function",
            "parts": [
                glm.Part(
                    function_response=glm.FunctionResponse(
                        name=response.parts[0].function_call.name,
                        response={"error": f"Validation Error(s) found:\n{exception}"},
                    )
                ),
            ],
        },
        {
            "role": "user",
            "parts": ["Recall the function arguments correctly and fix the errors"],
        },
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_gemini_json(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Gemini
    exception: Exception,
):
    """
    Handle reask for Gemini JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "contents" (user message requesting JSON correction)
    """
    kwargs["contents"].append(
        {
            "role": "user",
            "parts": [
                f"Correct the following JSON response, based on the errors given below:\n\n"
                f"JSON:\n{response.text}\n\nExceptions:\n{exception}"
            ],
        }
    )
    return kwargs


def reask_vertexai_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Vertex AI
    exception: Exception,
):
    """
    Handle reask for Vertex AI tools mode when validation fails.

    Kwargs modifications:
    - Adds: "contents" (tool response messages indicating validation errors)
    """
    from instructor.client_vertexai import vertexai_function_response_parser

    kwargs = kwargs.copy()
    reask_msgs = [
        response.candidates[0].content,
        vertexai_function_response_parser(response, exception),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_vertexai_json(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Vertex AI
    exception: Exception,
):
    """
    Handle reask for Vertex AI JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "contents" (user message requesting JSON correction)
    """
    from instructor.client_vertexai import vertexai_message_parser

    kwargs = kwargs.copy()

    reask_msgs = [
        response.candidates[0].content,
        vertexai_message_parser(
            {
                "role": "user",
                "content": (
                    f"Validation Errors found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors found in the following attempt:\n{response.text}"
                ),
            }
        ),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_genai_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Google GenAI tools mode when validation fails.

    Kwargs modifications:
    - Adds: "contents" (tool response messages indicating validation errors)
    """
    from google.genai import types

    kwargs = kwargs.copy()
    function_call = response.candidates[0].content.parts[0].function_call
    kwargs["contents"].append(
        types.ModelContent(
            parts=[
                types.Part.from_function_call(
                    name=function_call.name,
                    args=function_call.args,
                ),
                types.Part.from_text(
                    text=f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            ]
        ),
    )
    return kwargs


def reask_genai_structured_outputs(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Google GenAI structured outputs mode when validation fails.

    Kwargs modifications:
    - Adds: "contents" (user message describing validation errors)
    """
    from google.genai import types

    kwargs = kwargs.copy()

    genai_response = (
        response.text
        if response and hasattr(response, "text")
        else "You must generate a response to the user's request that is consistent with the response model"
    )

    kwargs["contents"].append(
        types.ModelContent(
            parts=[
                types.Part.from_text(
                    text=f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors in the following attempt:\n{genai_response}"
                ),
            ]
        ),
    )
    return kwargs


# Response handlers
def handle_genai_message_conversion(
    new_kwargs: dict[str, Any], autodetect_images: bool = False
) -> dict[str, Any]:
    """
    Convert OpenAI-style messages to GenAI contents.

    Kwargs modifications:
    - Removes: "messages"
    - Adds: "contents" (GenAI-style messages)
    - Adds: "config" (system instruction) when system not provided
    """
    from google.genai import types

    messages = new_kwargs.get("messages", [])

    # Convert OpenAI-style messages to GenAI-style contents
    new_kwargs["contents"] = convert_to_genai_messages(messages)

    # Extract multimodal content for GenAI
    from ...processing.multimodal import extract_genai_multimodal_content

    new_kwargs["contents"] = extract_genai_multimodal_content(
        new_kwargs["contents"], autodetect_images
    )

    # Handle system message for GenAI
    if "system" not in new_kwargs:
        system_message = extract_genai_system_message(messages)
        if system_message:
            new_kwargs["config"] = types.GenerateContentConfig(
                system_instruction=system_message
            )

    # Remove messages since we converted to contents
    new_kwargs.pop("messages", None)

    return new_kwargs


def handle_gemini_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Gemini JSON mode.

    When response_model is None:
        - Updates kwargs for Gemini compatibility (converts messages format)
        - No JSON schema or response format is configured

    When response_model is provided:
        - Adds/modifies system message with JSON schema instructions
        - Sets response_mime_type to "application/json"
        - Updates kwargs for Gemini compatibility

    Kwargs modifications:
    - Modifies: "messages" (adds/modifies system message with JSON schema) - only when response_model provided
    - Adds/Modifies: "generation_config" (sets response_mime_type to "application/json") - only when response_model provided
    - All modifications from update_gemini_kwargs (converts messages to Gemini format)
    """
    if "model" in new_kwargs:
        raise ConfigurationError(
            "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
        )

    if response_model is None:
        # Just handle message conversion
        new_kwargs = update_gemini_kwargs(new_kwargs)
        return None, new_kwargs

    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    if new_kwargs["messages"][0]["role"] != "system":
        new_kwargs["messages"].insert(0, {"role": "system", "content": message})
    else:
        new_kwargs["messages"][0]["content"] += f"\n\n{message}"

    new_kwargs["generation_config"] = new_kwargs.get("generation_config", {}) | {
        "response_mime_type": "application/json"
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_gemini_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Gemini tools mode.

    Kwargs modifications:
    - When response_model is None: Only applies update_gemini_kwargs transformations
    - When response_model is provided:
      - Adds: "tools" (list with gemini schema)
      - Adds: "tool_config" (function calling config with mode and allowed functions)
      - All modifications from update_gemini_kwargs
    """
    if "model" in new_kwargs:
        raise ConfigurationError(
            "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
        )

    if response_model is None:
        # Just handle message conversion
        new_kwargs = update_gemini_kwargs(new_kwargs)
        return None, new_kwargs

    new_kwargs["tools"] = [response_model.gemini_schema]
    new_kwargs["tool_config"] = {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [response_model.__name__],
        },
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_genai_structured_outputs(
    response_model: type[Any] | None,
    new_kwargs: dict[str, Any],
    autodetect_images: bool = False,
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Google GenAI structured outputs mode.

    Kwargs modifications:
    - When response_model is None: Applies handle_genai_message_conversion
    - When response_model is provided:
      - Removes: "messages", "response_model", "generation_config", "safety_settings"
      - Adds: "contents" (GenAI-style messages)
      - Adds: "config" (GenerateContentConfig with system_instruction, response_mime_type, response_schema)
      - Handles multimodal content extraction
    """
    from google.genai import types

    if response_model is None:
        # Just handle message conversion
        new_kwargs = handle_genai_message_conversion(new_kwargs, autodetect_images)
        return None, new_kwargs

    # Automatically wrap regular models with Partial when streaming is enabled
    if new_kwargs.get("stream", False) and not issubclass(response_model, PartialBase):
        response_model = Partial[response_model]

    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    # Extract multimodal content for GenAI
    from ...processing.multimodal import extract_genai_multimodal_content

    new_kwargs["contents"] = extract_genai_multimodal_content(
        new_kwargs["contents"], autodetect_images
    )

    # We validate that the schema doesn't contain any Union fields
    map_to_gemini_function_schema(response_model.model_json_schema())

    base_config = {
        "system_instruction": system_message,
        "response_mime_type": "application/json",
        "response_schema": response_model,
    }

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)
    new_kwargs.pop("thinking_config", None)

    return response_model, new_kwargs


def handle_genai_tools(
    response_model: type[Any] | None,
    new_kwargs: dict[str, Any],
    autodetect_images: bool = False,
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Google GenAI tools mode.

    Kwargs modifications:
    - When response_model is None: Applies handle_genai_message_conversion
    - When response_model is provided:
      - Removes: "messages", "response_model", "generation_config", "safety_settings"
      - Adds: "contents" (GenAI-style messages)
      - Adds: "config" (GenerateContentConfig with tools and tool_config)
      - Handles multimodal content extraction
    """
    from google.genai import types

    if response_model is None:
        # Just handle message conversion
        new_kwargs = handle_genai_message_conversion(new_kwargs, autodetect_images)
        return None, new_kwargs

    # Automatically wrap regular models with Partial when streaming is enabled
    if new_kwargs.get("stream", False) and not issubclass(response_model, PartialBase):
        response_model = Partial[response_model]

    schema = map_to_gemini_function_schema(response_model.model_json_schema())
    function_definition = types.FunctionDeclaration(
        name=response_model.__name__,
        description=response_model.__doc__,
        parameters=schema,
    )

    # We support the system message if you declare a system kwarg or if you pass a system message in the messages
    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    base_config = {
        "system_instruction": system_message,
        "tools": [types.Tool(function_declarations=[function_definition])],
        "tool_config": types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[response_model.__name__]
            ),
        ),
    }

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    # Extract multimodal content for GenAI
    from ...processing.multimodal import extract_genai_multimodal_content

    new_kwargs["contents"] = extract_genai_multimodal_content(
        new_kwargs["contents"], autodetect_images
    )

    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)
    new_kwargs.pop("thinking_config", None)

    return response_model, new_kwargs


def handle_vertexai_parallel_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """
    Handle Vertex AI parallel tools mode.

    Kwargs modifications:
    - Adds: "contents", "tools", "tool_config" via vertexai_process_response
    - Validates: stream=False
    """
    from typing import get_args

    from instructor.client_vertexai import vertexai_process_response
    from instructor.dsl.parallel import VertexAIParallelModel

    if new_kwargs.get("stream", False):
        raise ConfigurationError(
            "stream=True is not supported when using VERTEXAI_PARALLEL_TOOLS mode"
        )

    # Extract concrete types before passing to vertexai_process_response
    model_types = list(get_args(response_model))
    contents, tools, tool_config = vertexai_process_response(new_kwargs, model_types)
    new_kwargs["contents"] = contents
    new_kwargs["tools"] = tools
    new_kwargs["tool_config"] = tool_config

    return VertexAIParallelModel(typehint=response_model), new_kwargs


def handle_vertexai_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    from instructor.client_vertexai import vertexai_process_response

    """
    Handle Vertex AI tools mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "contents", "tools", "tool_config" via vertexai_process_response
    """

    if response_model is None:
        # Just handle message conversion - keep the messages as they are
        return None, new_kwargs

    contents, tools, tool_config = vertexai_process_response(new_kwargs, response_model)

    new_kwargs["contents"] = contents
    new_kwargs["tools"] = tools
    new_kwargs["tool_config"] = tool_config
    return response_model, new_kwargs


def handle_vertexai_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    from instructor.providers.vertexai.client import vertexai_process_json_response

    """
    Handle Vertex AI JSON mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "contents" and "generation_config" via vertexai_process_json_response
    """

    if response_model is None:
        # Just handle message conversion - keep the messages as they are
        return None, new_kwargs

    contents, generation_config = vertexai_process_json_response(
        new_kwargs, response_model
    )

    new_kwargs["contents"] = contents
    new_kwargs["generation_config"] = generation_config
    return response_model, new_kwargs


# Handler registry for Google providers
GOOGLE_HANDLERS = {
    Mode.GEMINI_TOOLS: {
        "reask": reask_gemini_tools,
        "response": handle_gemini_tools,
    },
    Mode.GEMINI_JSON: {
        "reask": reask_gemini_json,
        "response": handle_gemini_json,
    },
    Mode.GENAI_TOOLS: {
        "reask": reask_genai_tools,
        "response": handle_genai_tools,
    },
    Mode.GENAI_STRUCTURED_OUTPUTS: {
        "reask": reask_genai_structured_outputs,
        "response": handle_genai_structured_outputs,
    },
    Mode.VERTEXAI_TOOLS: {
        "reask": reask_vertexai_tools,
        "response": handle_vertexai_tools,
    },
    Mode.VERTEXAI_JSON: {
        "reask": reask_vertexai_json,
        "response": handle_vertexai_json,
    },
    Mode.VERTEXAI_PARALLEL_TOOLS: {
        "reask": reask_vertexai_tools,
        "response": handle_vertexai_parallel_tools,
    },
}

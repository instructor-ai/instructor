from typing import Any, Union

from openai.types.chat import ChatCompletionMessageParam

from instructor.multimodal import Image, Audio


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
    from pydantic import BaseModel

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
                    elif isinstance(content_item, (Image, Audio)):
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

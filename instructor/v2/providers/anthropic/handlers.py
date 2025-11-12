"""v2 Anthropic mode handlers using class-based pattern.

Each handler class implements prepare_request, handle_reask, and parse_response
methods, then registers via decorator.
"""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, TypeAdapter
from typing import Annotated

if TYPE_CHECKING:
    from anthropic.types import Message

from instructor import Mode, Provider
from instructor.core.exceptions import IncompleteOutputException
from instructor.processing.function_calls import extract_json_from_codeblock
from instructor.processing.multimodal import Image, Audio, PDF
from instructor.providers.anthropic.utils import (
    combine_system_messages,
    extract_system_messages,
    generate_anthropic_schema,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


def serialize_message_content(content: Any) -> Any:
    """Serialize message content, converting Pydantic models to dicts.

    Args:
        content: Message content (string, list, dict, or Pydantic model)

    Returns:
        Serialized content with Pydantic models converted to dicts
    """
    if isinstance(content, Image):
        # Convert Image object to Anthropic's expected format
        source = str(content.source)

        # Determine source type based on the source value
        if source.startswith(("http://", "https://")):
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": source,
                },
            }
        elif source.startswith("data:"):
            # Base64-encoded data URL
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.media_type,
                    "data": content.data or source.split(",")[1],
                },
            }
        else:
            # File path or base64 string
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.media_type,
                    "data": content.data or source,
                },
            }
    elif isinstance(content, PDF):
        # Convert PDF object to Anthropic's expected format
        source = str(content.source)

        if source.startswith(("http://", "https://")):
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": source,
                },
            }
        elif source.startswith("data:"):
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": content.data or source.split(",")[1],
                },
            }
        else:
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": content.data or source,
                },
            }
    elif isinstance(content, Audio):
        # Audio handling similar to Image
        source = str(content.source)

        if source.startswith(("http://", "https://")):
            return {
                "type": "audio",
                "source": {
                    "type": "url",
                    "url": source,
                },
            }
        else:
            return {
                "type": "audio",
                "source": {
                    "type": "base64",
                    "media_type": content.media_type,
                    "data": content.data or source,
                },
            }
    elif isinstance(content, str):
        # Convert plain text strings to Anthropic's text content format
        return {
            "type": "text",
            "text": content,
        }
    elif isinstance(content, list):
        # Process list content recursively
        return [serialize_message_content(item) for item in content]
    elif isinstance(content, dict):
        # Check if already in Anthropic format (has "type" key)
        if "type" in content:
            # Already formatted, just recurse on values
            return {k: serialize_message_content(v) for k, v in content.items()}
        # Plain dict, recurse on values
        return {k: serialize_message_content(v) for k, v in content.items()}
    elif hasattr(content, "model_dump"):
        # Handle any other Pydantic BaseModel
        return content.model_dump()
    else:
        return content


def process_messages_for_anthropic(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process messages to serialize any Pydantic models in content.

    Args:
        messages: List of message dicts

    Returns:
        Processed messages with serialized content
    """
    processed = []
    for message in messages:
        msg_copy = message.copy()
        if "content" in msg_copy:
            content = msg_copy["content"]
            # Only deeply process list content - convert strings/objects to proper format
            # If content is a string, leave it as-is (Anthropic accepts plain strings)
            # If content is a list, process each item
            if isinstance(content, list):
                msg_copy["content"] = serialize_message_content(content)
            elif isinstance(content, (Image, Audio, PDF)) or hasattr(
                content, "model_dump"
            ):
                # Serialize Pydantic models to dict
                msg_copy["content"] = serialize_message_content(content)
            # Leave strings as-is, and dicts with "type" key as-is
        processed.append(msg_copy)
    return processed


@register_mode_handler(Provider.ANTHROPIC, Mode.TOOLS)
class AnthropicToolsHandler(ModeHandler):
    """Handler for Anthropic TOOLS mode.

    Generates tool schemas, forces tool use, and parses tool_use blocks.
    """

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request kwargs for TOOLS mode.

        Args:
            response_model: Pydantic model to extract (or None)
            kwargs: Original request kwargs

        Returns:
            Tuple of (response_model, modified_kwargs)
        """
        new_kwargs = kwargs.copy()

        # Extract and combine system messages BEFORE serializing message content
        system_messages = extract_system_messages(new_kwargs.get("messages", []))

        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )

        # Remove system messages from messages list
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]

        # Serialize message content AFTER extracting system messages
        if "messages" in new_kwargs:
            new_kwargs["messages"] = process_messages_for_anthropic(
                new_kwargs["messages"]
            )

        if response_model is None:
            # Just return with processed messages and extracted system
            return None, new_kwargs

        # Generate tool schema
        tool_descriptions = generate_anthropic_schema(response_model)
        new_kwargs["tools"] = [tool_descriptions]
        new_kwargs["tool_choice"] = {
            "type": "tool",
            "name": response_model.__name__,
        }

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle validation failure for TOOLS mode.

        Args:
            kwargs: Original request kwargs
            response: Failed API response
            exception: Validation exception

        Returns:
            Modified kwargs for retry
        """
        kwargs = kwargs.copy()
        from anthropic.types import Message

        from instructor.core.exceptions import ResponseParsingError

        if not isinstance(response, Message):
            raise ResponseParsingError(
                "Response must be an Anthropic Message",
                mode="ANTHROPIC_TOOLS",
                raw_response=response,
            )

        # Extract assistant's response
        assistant_content = []
        tool_use_id = None
        for content in response.content:
            assistant_content.append(content.model_dump())
            if content.type == "tool_use":
                tool_use_id = content.id

        # Build reask messages
        reask_msgs = [{"role": "assistant", "content": assistant_content}]

        if tool_use_id is not None:
            # Tool was called, return error as tool_result
            reask_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
                            "is_error": True,
                        }
                    ],
                }
            )
        else:
            # No tool call, ask for correction
            reask_msgs.append(
                {
                    "role": "user",
                    "content": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors",
                }
            )

        kwargs["messages"].extend(reask_msgs)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Parse TOOLS mode response.

        Args:
            response: Anthropic API response
            response_model: Pydantic model to validate against
            validation_context: Optional context for validation
            strict: Optional strict validation mode

        Returns:
            Validated Pydantic model instance

        Raises:
            IncompleteOutputException: If response hit max_tokens
            ValidationError: If response doesn't match model
        """
        from anthropic.types import Message

        if isinstance(response, Message) and response.stop_reason == "max_tokens":
            raise IncompleteOutputException(last_completion=response)

        # Extract tool calls
        tool_calls = [
            json.dumps(c.input) for c in response.content if c.type == "tool_use"
        ]

        # Validate exactly one tool call
        tool_calls_validator = TypeAdapter(
            Annotated[list[Any], Field(min_length=1, max_length=1)]
        )
        tool_call = tool_calls_validator.validate_python(tool_calls)[0]

        parsed = response_model.model_validate_json(
            tool_call, context=validation_context, strict=strict
        )

        # Attach raw response for access via create_with_completion
        parsed._raw_response = response  # type: ignore

        return parsed


@register_mode_handler(Provider.ANTHROPIC, Mode.JSON)
class AnthropicJSONHandler(ModeHandler):
    """Handler for Anthropic JSON mode.

    Injects JSON schema into system message and parses JSON from text blocks.
    """

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request kwargs for JSON mode.

        Args:
            response_model: Pydantic model to extract (or None)
            kwargs: Original request kwargs

        Returns:
            Tuple of (response_model, modified_kwargs)
        """
        new_kwargs = kwargs.copy()

        # Extract and combine system messages BEFORE serializing message content
        system_messages = extract_system_messages(new_kwargs.get("messages", []))

        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )

        # Remove system messages from messages list
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m["role"] != "system"
        ]

        # Serialize message content AFTER extracting system messages
        if "messages" in new_kwargs:
            new_kwargs["messages"] = process_messages_for_anthropic(
                new_kwargs["messages"]
            )

        if response_model is None:
            return None, new_kwargs

        # Add JSON schema to system message
        json_schema_message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"),
            [{"type": "text", "text": json_schema_message}],
        )

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Message,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle validation failure for JSON mode.

        Args:
            kwargs: Original request kwargs
            response: Failed API response
            exception: Validation exception

        Returns:
            Modified kwargs for retry
        """
        kwargs = kwargs.copy()
        from anthropic.types import Message

        from instructor.core.exceptions import ResponseParsingError

        if not isinstance(response, Message):
            raise ResponseParsingError(
                "Response must be an Anthropic Message",
                mode="JSON",
                raw_response=response,
            )

        # Filter for text blocks to handle ThinkingBlock and other non-text content
        text_blocks = [c for c in response.content if c.type == "text"]
        if not text_blocks:
            text_content = "No text content found in response"
        else:
            # Use the last text block
            text_content = text_blocks[-1].text

        reask_msg = {
            "role": "user",
            "content": f"""Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{text_content}""",
        }
        kwargs["messages"].append(reask_msg)
        return kwargs

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> BaseModel:
        """Parse JSON mode response.

        Args:
            response: Anthropic API response
            response_model: Pydantic model to validate against
            validation_context: Optional context for validation
            strict: Optional strict validation mode

        Returns:
            Validated Pydantic model instance

        Raises:
            IncompleteOutputException: If response hit max_tokens
            ValidationError: If response doesn't match model
        """
        from anthropic.types import Message

        if hasattr(response, "choices"):
            # Handle OpenAI-style response (shouldn't happen for Anthropic)
            completion = response.choices[0]
            if completion.finish_reason == "length":
                raise IncompleteOutputException(last_completion=completion)
            text = completion.message.content
        else:
            from instructor.core.exceptions import ResponseParsingError

            if not isinstance(response, Message):
                raise ResponseParsingError(
                    "Response must be an Anthropic Message",
                    mode="JSON",
                    raw_response=response,
                )
            if response.stop_reason == "max_tokens":
                raise IncompleteOutputException(last_completion=response)

            # Find the last text block
            text_blocks = [c for c in response.content if c.type == "text"]
            last_block = text_blocks[-1]

            # Strip raw control characters (0x00-0x1F)
            text = re.sub(r"[\u0000-\u001F]", "", last_block.text)

        # Extract JSON from potential code block
        extra_text = extract_json_from_codeblock(text)

        if strict:
            parsed = response_model.model_validate_json(
                extra_text, context=validation_context, strict=strict
            )
        else:
            parsed = response_model.model_validate_json(
                extra_text, context=validation_context
            )

        # Attach raw response for access via create_with_completion
        parsed._raw_response = response  # type: ignore

        return parsed

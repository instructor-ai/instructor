# type: ignore[all]
from __future__ import annotations
from typing import Any
from textwrap import dedent
from instructor.mode import Mode
from jinja2.sandbox import SandboxedEnvironment


def apply_template(text: str, context: dict[str, Any]) -> str:
    """Apply Jinja2 template to the given text."""
    return dedent(SandboxedEnvironment().from_string(text).render(**context))


def process_message(
    message: dict[str, Any], context: dict[str, Any], mode: Mode
) -> dict[str, Any]:
    """Process a single message, applying templates to its content."""
    if mode in {Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS}:
        from google.genai import types

        return types.Content(
            role=message.role,
            parts=[
                types.Part.from_text(text=apply_template(part.text, context))
                if hasattr(part, "text")
                else part
                for part in message.parts
            ],
        )

    # VertexAI Support
    if (
        hasattr(message, "parts")
        and isinstance(message.parts, list)
        and len(message.parts) > 0
        and not isinstance(message.parts[0], str)
    ):
        import vertexai.generative_models as gm

        return gm.Content(
            role=message.role,
            parts=[
                (
                    gm.Part.from_text(apply_template(part.text, context))
                    if hasattr(part, "text")
                    else part
                )
                for part in message.parts
            ],
        )

    # OpenAI format
    if isinstance(message.get("content"), str):
        message["content"] = apply_template(message["content"], context)
        return message

    # Anthropic format
    if isinstance(message.get("content"), list):
        for part in message["content"]:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = apply_template(part["text"], context)
        return message

    # Gemini Support
    if isinstance(message.get("parts"), list):
        message["parts"] = [
            apply_template(part, context) if isinstance(part, str) else part
            for part in message["parts"]
        ]
        return message

    # Cohere format
    if isinstance(message.get("message"), str):
        message["message"] = apply_template(message["message"], context)
        return message


def handle_templating(
    kwargs: dict[str, Any], mode: Mode, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Handle templating for messages using the provided context.

    This function processes messages, applying Jinja2 templating to their content
    using the provided context. It supports various message formats including
    OpenAI, Anthropic, Cohere, VertexAI, and Gemini.

    Args:
        kwargs (Dict[str, Any]): Keyword arguments being passed to the create method.
        context (Dict[str, Any] | None, optional): Variables to use in templating. Defaults to None.

    Returns:
        Dict[str, Any]: The processed kwargs with templated content.

    Raises:
        ValueError: If no recognized message format is found in kwargs.
    """
    if not context:
        return kwargs

    new_kwargs = kwargs.copy()

    if mode in {Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS} and "system" in new_kwargs:
        new_kwargs["system"] = apply_template(new_kwargs["system"], context)

    # Handle Cohere's message field
    if "message" in new_kwargs:
        new_kwargs["message"] = apply_template(new_kwargs["message"], context)
        new_kwargs["chat_history"] = [
            process_message(message, context, mode)
            for message in new_kwargs["chat_history"]
        ]

        return new_kwargs

    if isinstance(new_kwargs, list):
        messages = new_kwargs
        if not messages:
            return
    elif isinstance(new_kwargs, dict):
        messages = new_kwargs.get("messages") or new_kwargs.get("contents")

    if not messages:
        return

    if "messages" in new_kwargs:
        if mode in {Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS}:
            templated_messages = []
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "system":
                    templated_msg = message.copy()
                    if isinstance(message.get("content"), str):
                        templated_msg["content"] = apply_template(message["content"], context)
                    elif isinstance(message.get("content"), list):
                        templated_content = []
                        for item in message["content"]:
                            if isinstance(item, str):
                                templated_content.append(apply_template(item, context))
                            else:
                                templated_content.append(item)
                        templated_msg["content"] = templated_content
                    templated_messages.append(templated_msg)
                else:
                    templated_messages.append(process_message(message, context, mode))
            new_kwargs["messages"] = templated_messages
        else:
            new_kwargs["messages"] = [
                process_message(message, context, mode) for message in messages
            ]

    elif "contents" in new_kwargs:
        new_kwargs["contents"] = [
            process_message(content, context, mode)
            for content in new_kwargs["contents"]
        ]

    return new_kwargs

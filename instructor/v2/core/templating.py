# type: ignore[all]
from __future__ import annotations
from typing import Any
from textwrap import dedent
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider, provider_from_mode
from jinja2.sandbox import SandboxedEnvironment


def apply_template(text: str, context: dict[str, Any]) -> str:
    """Apply Jinja2 template to the given text."""
    return dedent(SandboxedEnvironment().from_string(text).render(**context))


def process_message(
    message: dict[str, Any], context: dict[str, Any], provider: Provider
) -> dict[str, Any]:
    """Process a single message, applying templates to its content."""
    if provider == Provider.GENAI:
        from instructor.v2.providers.genai.templating import (
            process_message as process_genai_message,
        )

        return process_genai_message(message, context, apply_template)

    # VertexAI Support
    if (
        hasattr(message, "parts")
        and isinstance(message.parts, list)
        and len(message.parts) > 0
        and not isinstance(message.parts[0], str)
    ):
        from instructor.v2.providers.vertexai.templating import (
            process_message as process_vertexai_message,
        )

        return process_vertexai_message(message, context, apply_template)

    # OpenAI format
    if isinstance(message.get("content"), str):
        from instructor.v2.providers.openai.templating import (
            process_message as process_openai_message,
        )

        return process_openai_message(message, context, apply_template)

    # Anthropic format
    if isinstance(message.get("content"), list):
        from instructor.v2.providers.anthropic.templating import (
            process_message as process_anthropic_message,
        )

        return process_anthropic_message(message, context, apply_template)

    # Gemini Support
    if isinstance(message.get("parts"), list):
        from instructor.v2.providers.gemini.templating import (
            process_message as process_gemini_message,
        )

        return process_gemini_message(message, context, apply_template)

    # Cohere format
    if isinstance(message.get("message"), str):
        from instructor.v2.providers.cohere.templating import (
            process_message as process_cohere_message,
        )

        return process_cohere_message(message, context, apply_template)

    return message


def handle_templating(
    kwargs: dict[str, Any],
    mode: Mode,  # noqa: ARG001
    provider: Provider | dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
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
    if context is None and isinstance(provider, dict):
        context = provider
        provider = None

    if not context:
        return kwargs

    if provider is None:
        provider = provider_from_mode(mode, Provider.OPENAI)

    new_kwargs = kwargs.copy()

    # Handle Cohere's message field
    if "message" in new_kwargs:
        new_kwargs["message"] = apply_template(new_kwargs["message"], context)
        new_kwargs["chat_history"] = [
            process_message(message, context, provider)
            for message in new_kwargs["chat_history"]
        ]

        return new_kwargs

    if isinstance(new_kwargs, list):
        messages = new_kwargs
        if not messages:
            return new_kwargs
    elif isinstance(new_kwargs, dict):
        messages = new_kwargs.get("messages") or new_kwargs.get("contents")

    if not messages:
        return new_kwargs

    if "messages" in new_kwargs:
        new_kwargs["messages"] = [
            process_message(message, context, provider) for message in messages
        ]

    elif "contents" in new_kwargs:
        new_kwargs["contents"] = [
            process_message(content, context, provider)
            for content in new_kwargs["contents"]
        ]

    return new_kwargs

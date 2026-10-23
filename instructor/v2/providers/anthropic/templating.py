"""Anthropic-specific message templating helpers."""

from __future__ import annotations

from typing import Any, Callable


def process_message(
    message: dict[str, Any],
    context: dict[str, Any],
    apply_template: Callable[[str, dict[str, Any]], str],
) -> dict[str, Any]:
    """Apply templates to Anthropic message content."""
    if isinstance(message.get("content"), list):
        for part in message["content"]:
            if (
                isinstance(part, dict)
                # List content from OpenAI Responses mode uses "input_text" parts.
                and part.get("type") in ("text", "input_text")
                and isinstance(part.get("text"), str)
            ):
                part["text"] = apply_template(part["text"], context)
    return message

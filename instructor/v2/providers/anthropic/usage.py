"""Anthropic-specific usage helpers."""

from __future__ import annotations

from typing import Any


def initialize_usage() -> Any:
    """Create an empty Anthropic usage accumulator."""
    from anthropic.types import Usage as AnthropicUsage

    return AnthropicUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def update_total_usage(response_usage: Any, total_usage: Any) -> bool:
    """Accumulate Anthropic token usage into a running total when applicable.

    Every numeric field is summed generically, including the leaves of nested
    sub-models such as ``cache_creation`` (``ephemeral_*_input_tokens``),
    ``server_tool_use`` (``web_search_requests`` / ``web_fetch_requests``) and
    ``output_tokens_details`` (``thinking_tokens``). Billable counters added by
    newer ``anthropic`` SDK releases are therefore picked up automatically
    instead of being left stale at whatever the last attempt reported.
    """
    from anthropic.types import Usage as AnthropicUsage

    if not isinstance(response_usage, AnthropicUsage) or not isinstance(
        total_usage, AnthropicUsage
    ):
        return False

    # Imported lazily to avoid a circular import at module load time.
    from instructor.v2.core.usage import accumulate_usage

    accumulate_usage(total_usage, response_usage)
    return True

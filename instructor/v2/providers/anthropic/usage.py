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


def _ensure_sub_models(total_usage: Any) -> None:
    """Ensure nested usage sub-models exist on the accumulator, zeroed."""
    from anthropic.types.usage import (
        CacheCreation,
        ServerToolUsage,
        OutputTokensDetails,
    )

    if not total_usage.cache_creation:
        total_usage.cache_creation = CacheCreation(
            ephemeral_1h_input_tokens=0,
            ephemeral_5m_input_tokens=0,
        )
    if not total_usage.server_tool_use:
        total_usage.server_tool_use = ServerToolUsage(
            web_search_requests=0,
            web_fetch_requests=0,
        )
    if not total_usage.output_tokens_details:
        total_usage.output_tokens_details = OutputTokensDetails(
            thinking_tokens=0,
        )


def update_total_usage(response_usage: Any, total_usage: Any) -> bool:
    """Accumulate Anthropic token usage into a running total when applicable."""
    from anthropic.types import Usage as AnthropicUsage

    if not isinstance(response_usage, AnthropicUsage) or not isinstance(
        total_usage, AnthropicUsage
    ):
        return False

    if not total_usage.cache_creation_input_tokens:
        total_usage.cache_creation_input_tokens = 0
    if not total_usage.cache_read_input_tokens:
        total_usage.cache_read_input_tokens = 0

    # Ensure nested sub-models exist on the accumulator before summing.
    _ensure_sub_models(total_usage)

    total_usage.input_tokens += response_usage.input_tokens or 0
    total_usage.output_tokens += response_usage.output_tokens or 0
    total_usage.cache_creation_input_tokens += (
        response_usage.cache_creation_input_tokens or 0
    )
    total_usage.cache_read_input_tokens += response_usage.cache_read_input_tokens or 0

    # Accumulate cache_creation sub-model fields.
    if resp_cc := response_usage.cache_creation:
        total_usage.cache_creation.ephemeral_1h_input_tokens = (
            (total_usage.cache_creation.ephemeral_1h_input_tokens or 0)
            + (resp_cc.ephemeral_1h_input_tokens or 0)
        )
        total_usage.cache_creation.ephemeral_5m_input_tokens = (
            (total_usage.cache_creation.ephemeral_5m_input_tokens or 0)
            + (resp_cc.ephemeral_5m_input_tokens or 0)
        )

    # Accumulate server_tool_use sub-model fields.
    if resp_stu := response_usage.server_tool_use:
        total_usage.server_tool_use.web_search_requests = (
            (total_usage.server_tool_use.web_search_requests or 0)
            + (resp_stu.web_search_requests or 0)
        )
        total_usage.server_tool_use.web_fetch_requests = (
            (total_usage.server_tool_use.web_fetch_requests or 0)
            + (resp_stu.web_fetch_requests or 0)
        )

    # Accumulate output_tokens_details sub-model fields.
    if resp_otd := response_usage.output_tokens_details:
        total_usage.output_tokens_details.thinking_tokens = (
            (total_usage.output_tokens_details.thinking_tokens or 0)
            + (resp_otd.thinking_tokens or 0)
        )

    # Write back the accumulated totals onto the response so callers see them.
    response_usage.input_tokens = total_usage.input_tokens
    response_usage.output_tokens = total_usage.output_tokens
    response_usage.cache_creation_input_tokens = total_usage.cache_creation_input_tokens
    response_usage.cache_read_input_tokens = total_usage.cache_read_input_tokens
    response_usage.cache_creation = total_usage.cache_creation.model_copy(deep=True)
    response_usage.server_tool_use = total_usage.server_tool_use.model_copy(deep=True)
    response_usage.output_tokens_details = total_usage.output_tokens_details.model_copy(
        deep=True
    )
    return True

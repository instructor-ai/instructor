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
    """Ensure nested usage sub-models exist on the accumulator, zeroed.

    ``OutputTokensDetails`` is guarded because it does not exist on the
    pinned ``anthropic==0.93.0`` — it was added in a later SDK release.
    """
    from anthropic.types.usage import CacheCreation, ServerToolUsage

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
    # output_tokens_details does not exist on anthropic==0.93.0 (pinned).
    # Use hasattr so this does not raise AttributeError on older SDKs.
    if not getattr(total_usage, "output_tokens_details", None):
        try:
            from anthropic.types.usage import OutputTokensDetails
        except ImportError:
            # anthropic<0.94.0 — field does not exist on the pinned version.
            return
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
    resp_otd = getattr(response_usage, "output_tokens_details", None)
    if resp_otd is not None:
        total_otd = getattr(total_usage, "output_tokens_details", None)
        if total_otd is not None:
            total_otd.thinking_tokens = (
                (total_otd.thinking_tokens or 0)
                + (resp_otd.thinking_tokens or 0)
            )

    # Write back the accumulated totals onto the response so callers see them.
    response_usage.input_tokens = total_usage.input_tokens
    response_usage.output_tokens = total_usage.output_tokens
    response_usage.cache_creation_input_tokens = total_usage.cache_creation_input_tokens
    response_usage.cache_read_input_tokens = total_usage.cache_read_input_tokens
    # Merge sub-models into the response rather than replacing, so fields the
    # accumulator does not track (e.g. future SDK additions) are preserved.
    if total_usage.cache_creation is not None:
        if response_usage.cache_creation is None:
            response_usage.cache_creation = total_usage.cache_creation.model_copy(deep=True)
        else:
            for field in ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"):
                val = getattr(total_usage.cache_creation, field, None)
                if val is not None:
                    setattr(response_usage.cache_creation, field, val)
    if total_usage.server_tool_use is not None:
        if response_usage.server_tool_use is None:
            response_usage.server_tool_use = total_usage.server_tool_use.model_copy(deep=True)
        else:
            for field in ("web_search_requests", "web_fetch_requests"):
                val = getattr(total_usage.server_tool_use, field, None)
                if val is not None:
                    setattr(response_usage.server_tool_use, field, val)
    total_otd = getattr(total_usage, "output_tokens_details", None)
    if total_otd is not None:
        resp_otd = getattr(response_usage, "output_tokens_details", None)
        if resp_otd is None:
            response_usage.output_tokens_details = total_otd.model_copy(deep=True)
        else:
            val = getattr(total_otd, "thinking_tokens", None)
            if val is not None:
                resp_otd.thinking_tokens = val
    return True

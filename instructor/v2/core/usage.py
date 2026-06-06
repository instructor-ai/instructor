"""Usage accumulation helpers owned by the v2 runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from openai.types import CompletionUsage as OpenAIUsage

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage

logger = logging.getLogger("instructor")
T_Response = TypeVar("T_Response")


def update_total_usage(
    response: T_Response | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Response | None:
    if response is None:
        return None

    response_usage = getattr(response, "usage", None)
    if isinstance(response_usage, OpenAIUsage) and isinstance(total_usage, OpenAIUsage):
        total_usage.completion_tokens += response_usage.completion_tokens or 0
        total_usage.prompt_tokens += response_usage.prompt_tokens or 0
        total_usage.total_tokens += response_usage.total_tokens or 0
        if (rtd := response_usage.completion_tokens_details) and (
            ttd := total_usage.completion_tokens_details
        ):
            ttd.audio_tokens = (ttd.audio_tokens or 0) + (rtd.audio_tokens or 0)
            ttd.reasoning_tokens = (ttd.reasoning_tokens or 0) + (
                rtd.reasoning_tokens or 0
            )
        if (rpd := response_usage.prompt_tokens_details) and (
            tpd := total_usage.prompt_tokens_details
        ):
            tpd.audio_tokens = (tpd.audio_tokens or 0) + (rpd.audio_tokens or 0)
            tpd.cached_tokens = (tpd.cached_tokens or 0) + (rpd.cached_tokens or 0)
        response_usage.completion_tokens = total_usage.completion_tokens
        response_usage.prompt_tokens = total_usage.prompt_tokens
        response_usage.total_tokens = total_usage.total_tokens
        response_usage.completion_tokens_details = (
            total_usage.completion_tokens_details.model_copy(deep=True)
            if total_usage.completion_tokens_details is not None
            else None
        )
        response_usage.prompt_tokens_details = (
            total_usage.prompt_tokens_details.model_copy(deep=True)
            if total_usage.prompt_tokens_details is not None
            else None
        )
        return response

    try:
        from instructor.v2.providers.anthropic.usage import (
            update_total_usage as update_anthropic_total_usage,
        )

        if update_anthropic_total_usage(response_usage, total_usage):
            return response
    except ImportError:
        pass

    logger.debug("No compatible response.usage found, token usage not updated.")
    return response

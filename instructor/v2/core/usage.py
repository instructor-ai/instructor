"""Usage accumulation helpers owned by the v2 runtime.

Accumulating token usage across retries used to enumerate the fields to sum by
hand, once per provider. Every counter a provider SDK added afterwards was then
silently dropped -- left stale (Anthropic) or overwritten with ``None`` when the
accumulator's details object was copied back onto the response (OpenAI). The
counters people check first (``input_tokens`` / ``prompt_tokens``) stayed
correct, so the under-count only surfaced when reconciling against an invoice.

The accumulator below instead discovers fields from the model and sums them
generically -- including the numeric leaves of nested sub-models such as
``cache_creation``, ``server_tool_use`` and ``*_tokens_details`` -- so new
billable fields are picked up automatically rather than needing a matching edit
in two places every time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage
    from openai.types import CompletionUsage as OpenAIUsage

logger = logging.getLogger("instructor")
T_Response = TypeVar("T_Response")


def _zero_numeric(model: BaseModel) -> None:
    """Set every numeric leaf of ``model`` to 0, recursing into sub-models."""
    for name in type(model).model_fields:
        value = getattr(model, name, None)
        if isinstance(value, BaseModel):
            _zero_numeric(value)
        elif isinstance(value, bool):
            continue  # bool is an int subclass; treat as a flag, not a counter
        elif isinstance(value, (int, float)):
            setattr(model, name, 0)


def _zeroed_copy(model: BaseModel) -> BaseModel:
    """A deep copy of ``model`` with all numeric leaves zeroed."""
    clone = model.model_copy(deep=True)
    _zero_numeric(clone)
    return clone


def _accumulate_into(total: BaseModel, response: BaseModel) -> None:
    """Add the numeric fields of ``response`` into ``total``, in place.

    Numeric fields (and the numeric leaves of nested sub-models) are summed;
    non-numeric fields (``service_tier``, ``inference_geo``, ...) take the
    latest reported value; a field that ``response`` reports as ``None`` leaves
    the running total untouched.
    """
    for name in type(response).model_fields:
        value = getattr(response, name, None)
        if isinstance(value, BaseModel):
            current = getattr(total, name, None)
            if not isinstance(current, BaseModel):
                # First attempt to report this sub-model: adopt a zeroed copy so
                # the response's own numbers are counted rather than skipped
                # (skipping them is what left the accumulator's field ``None``).
                current = _zeroed_copy(value)
                setattr(total, name, current)
            _accumulate_into(current, value)
        elif isinstance(value, bool):
            setattr(total, name, value)
        elif isinstance(value, (int, float)):
            running = getattr(total, name, None)
            setattr(total, name, (running or 0) + value)
        elif value is not None:
            setattr(total, name, value)


def _sync_into(response: BaseModel, total: BaseModel) -> None:
    """Mirror ``total`` back onto ``response``, in place.

    Existing (sub-)model instances on ``response`` are mutated rather than
    replaced, so a provider-specific ``Usage`` subclass keeps its type.
    """
    for name in type(total).model_fields:
        value = getattr(total, name, None)
        if isinstance(value, BaseModel):
            existing = getattr(response, name, None)
            if isinstance(existing, BaseModel):
                _sync_into(existing, value)
            else:
                setattr(response, name, value.model_copy(deep=True))
        else:
            setattr(response, name, value)


def accumulate_usage(total: BaseModel, response_usage: BaseModel) -> None:
    """Accumulate ``response_usage`` into ``total`` and mirror the running total
    back onto ``response_usage`` (both mutated in place).

    Works for any pydantic usage model -- the OpenAI ``CompletionUsage`` and the
    Anthropic ``Usage`` are both handled by the same generic walk.
    """
    _accumulate_into(total, response_usage)
    _sync_into(response_usage, total)


def update_total_usage(
    response: T_Response | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Response | None:
    if response is None:
        return None

    from openai.types import CompletionUsage as _OpenAIUsage

    response_usage = getattr(response, "usage", None)
    if isinstance(response_usage, _OpenAIUsage) and isinstance(
        total_usage, _OpenAIUsage
    ):
        accumulate_usage(total_usage, response_usage)
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

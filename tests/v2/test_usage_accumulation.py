"""Regression tests for retry usage accumulation (issue #2493).

Retries used to sum a hand-written list of token fields, so nested billable
counters were either left stale (Anthropic) or wiped to ``None`` when the
accumulator's details object was copied back onto the response (OpenAI). These
tests reproduce the reported accounting bugs and assert the generic accumulator
now sums them. All of it is pure accounting logic -- no API keys, no network.
"""

from __future__ import annotations

from pydantic import BaseModel

from instructor.v2.core.usage import accumulate_usage


def test_anthropic_accumulates_nested_billable_fields() -> None:
    from anthropic.types import Usage
    from anthropic.types.cache_creation import CacheCreation
    from anthropic.types.server_tool_usage import ServerToolUsage

    from instructor.v2.providers.anthropic.usage import (
        initialize_usage,
        update_total_usage,
    )

    total = initialize_usage()
    response: Usage | None = None
    for _ in range(3):
        response = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=0,
            cache_creation=CacheCreation(
                ephemeral_5m_input_tokens=500,
                ephemeral_1h_input_tokens=0,
            ),
            server_tool_use=ServerToolUsage(
                web_search_requests=2,
                web_fetch_requests=0,
            ),
        )
        assert update_total_usage(response, total) is True

    # Flat counters were already cumulative before the fix.
    assert total.input_tokens == 300
    assert total.output_tokens == 150
    assert total.cache_creation_input_tokens == 1500
    # Nested sub-models used to be frozen at the last attempt's value.
    assert total.cache_creation.ephemeral_5m_input_tokens == 1500
    assert total.server_tool_use.web_search_requests == 6
    # The response is mirrored to the running total.
    assert response is not None
    assert response.input_tokens == 300
    assert response.cache_creation.ephemeral_5m_input_tokens == 1500
    assert response.server_tool_use.web_search_requests == 6


def test_openai_accumulates_and_does_not_wipe_detail_fields() -> None:
    from openai.types import CompletionUsage
    from openai.types.completion_usage import (
        CompletionTokensDetails,
        PromptTokensDetails,
    )

    # Mirrors how the v2 retry runtime seeds the accumulator: the details
    # objects only carry the two fields it happened to enumerate.
    total = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    )
    response: CompletionUsage | None = None
    for _ in range(3):
        response = CompletionUsage(
            completion_tokens=50,
            prompt_tokens=100,
            total_tokens=150,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=40,
                accepted_prediction_tokens=7,
                rejected_prediction_tokens=3,
            ),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=500),
        )
        accumulate_usage(total, response)

    assert total.prompt_tokens == 300
    assert total.completion_tokens == 150
    assert total.total_tokens == 450
    assert total.completion_tokens_details.reasoning_tokens == 120
    # These were overwritten with None before the fix (accumulator never
    # populated them, then its details object was copied onto the response).
    assert total.completion_tokens_details.accepted_prediction_tokens == 21
    assert total.completion_tokens_details.rejected_prediction_tokens == 9
    assert total.prompt_tokens_details.cached_tokens == 1500
    assert response is not None
    assert response.completion_tokens_details.accepted_prediction_tokens == 21


def test_openai_usage_subclass_type_is_preserved() -> None:
    # A provider may hand back a CompletionUsage subclass; mirroring the total
    # must not reconstruct it as the base class.
    from openai.types import CompletionUsage

    class ProviderUsage(CompletionUsage):
        pass

    total = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    response = ProviderUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)

    accumulate_usage(total, response)

    assert isinstance(response, ProviderUsage)
    assert response.prompt_tokens == 16
    assert response.completion_tokens == 10
    assert response.total_tokens == 26


def test_generic_engine_handles_nesting_none_and_non_numeric() -> None:
    # Version-independent proof of the accumulation rules on a synthetic model.
    class Details(BaseModel):
        a: int | None = None
        b: int | None = None

    class Usage(BaseModel):
        x: int = 0
        tier: str | None = None
        details: Details | None = None

    total = Usage()
    # First attempt: accumulator's sub-model is None and must be adopted zeroed.
    accumulate_usage(total, Usage(x=10, tier="standard", details=Details(a=1, b=2)))
    # Second attempt: b is None and must not reset the running total.
    accumulate_usage(total, Usage(x=5, tier="priority", details=Details(a=3)))

    assert total.x == 15
    assert total.tier == "priority"  # non-numeric: latest value wins
    assert total.details is not None
    assert total.details.a == 4
    assert total.details.b == 2  # a None report left the prior total intact

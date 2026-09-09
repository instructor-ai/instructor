"""Offline SDK contracts and reproductions for the September 2026 usage audit.

Tests named ``current`` characterize limitations, not a promised future API.
They should change alongside an approved fix. No transports or mocks are used.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseUsage
from pydantic import BaseModel

from instructor.batch import BatchProcessor
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import _initialize_usage, _usage_total_tokens
from instructor.v2.core.usage import has_compatible_usage, update_total_usage
from instructor.v2.providers.openai.handlers import OpenAIJSONHandler


def completion(usage: CompletionUsage) -> ChatCompletion:
    return ChatCompletion(
        id="usage-audit",
        created=0,
        model="audit",
        object="chat.completion",
        choices=[],
        usage=usage,
    )


def test_current_retry_usage_mutates_response_and_manufactures_details() -> None:
    response = completion(
        CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert response.usage is not None
    assert response.usage.prompt_tokens_details is None
    total = _initialize_usage(Provider.OPENAI)
    update_total_usage(response, total)
    assert response.usage.prompt_tokens_details.cached_tokens == 0
    second = completion(
        CompletionUsage(prompt_tokens=20, completion_tokens=7, total_tokens=27)
    )
    update_total_usage(second, total)
    assert second.usage is not None
    assert second.usage.total_tokens == 42  # Provider supplied 27 for this attempt.
    assert response.usage.total_tokens == 15


def test_current_nested_dictionary_cost_is_last_attempt_only() -> None:
    total = _initialize_usage(Provider.OPENAI)
    for cost in [0.25, 0.75]:
        usage = CompletionUsage.model_validate(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": cost,
                "cost_details": {"upstream_inference_cost": cost},
            }
        )
        update_total_usage(completion(usage), total)
    assert total.model_extra["cost"] == 1.0
    assert total.model_extra["cost_details"]["upstream_inference_cost"] == 0.75


def test_anthropic_thinking_detail_accumulates_across_sdk_shapes() -> None:
    anthropic = pytest.importorskip("anthropic.types")
    if "output_tokens_details" in anthropic.Usage.model_fields:
        pytest.skip(
            "Newer SDK declares this field; audit the installed shape separately"
        )
    total = _initialize_usage(Provider.ANTHROPIC)
    for _ in range(2):
        message = anthropic.Message.model_validate(
            {
                "id": "audit",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "audit",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "output_tokens_details": {"thinking_tokens": 12},
                },
            }
        )
        update_total_usage(message, total)
    assert total.output_tokens == 40
    assert total.model_dump()["output_tokens_details"]["thinking_tokens"] == 24
    assert message.usage.model_dump()["output_tokens_details"]["thinking_tokens"] == 24


def test_openai_subsets_are_not_added_again_to_total() -> None:
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80, "audio_tokens": 10},
            "completion_tokens_details": {
                "reasoning_tokens": 30,
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 3,
            },
            "is_billable": True,
        }
    )
    total = _initialize_usage(Provider.OPENAI)
    for _ in range(2):
        update_total_usage(completion(usage.model_copy(deep=True)), total)
    assert _usage_total_tokens(total) == 300
    assert total.prompt_tokens_details.cached_tokens == 160
    assert total.completion_tokens_details.reasoning_tokens == 60
    assert total.model_extra["is_billable"] is True


def test_anthropic_cache_ttl_and_tools_have_separate_semantics() -> None:
    anthropic = pytest.importorskip("anthropic.types")
    total = _initialize_usage(Provider.ANTHROPIC)
    for _ in range(2):
        message = anthropic.Message.model_validate(
            {
                "id": "audit",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "audit",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 30,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 20,
                        "ephemeral_1h_input_tokens": 10,
                    },
                    "server_tool_use": {
                        "web_search_requests": 2,
                        "web_fetch_requests": 1,
                    },
                },
            }
        )
        update_total_usage(message, total)
    assert _usage_total_tokens(total) == 760
    assert total.cache_creation.ephemeral_1h_input_tokens == 20
    assert total.server_tool_use.web_search_requests == 4


def test_current_responses_api_usage_is_not_recognized() -> None:
    response = Response.model_validate(
        {
            "id": "resp_audit",
            "created_at": 0,
            "object": "response",
            "model": "audit",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 8},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        }
    )
    assert isinstance(response.usage, ResponseUsage)
    total = _initialize_usage(Provider.OPENAI)
    assert not has_compatible_usage(response, total)
    update_total_usage(response, total)
    assert total.total_tokens == 0
    assert response.usage.total_tokens == 15


def test_current_groq_sdk_usage_is_not_recognized() -> None:
    groq = pytest.importorskip("groq.types.chat")
    response = groq.ChatCompletion.model_validate(
        {
            "id": "audit",
            "created": 0,
            "object": "chat.completion",
            "model": "audit",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "queue_time": 0.1,
            },
        }
    )
    assert not has_compatible_usage(response, _initialize_usage(Provider.GROQ))


def usage_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="audit",
        created=0,
        model="audit",
        object="chat.completion.chunk",
        choices=[],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_current_stream_extractor_discards_usage_only_chunk() -> None:
    handler = OpenAIJSONHandler()
    assert list(handler.extract_streaming_json([usage_chunk()])) == []


@pytest.mark.asyncio
async def test_current_async_stream_extractor_discards_usage_only_chunk() -> None:
    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        yield usage_chunk()

    handler = OpenAIJSONHandler()
    assert [
        chunk async for chunk in handler.extract_streaming_json_async(chunks())
    ] == []


def test_current_batch_success_drops_per_item_usage() -> None:
    class Item(BaseModel):
        value: int

    processor = BatchProcessor("openai/audit", Item)
    raw = {
        "custom_id": "item",
        "response": {
            "body": {
                "choices": [{"message": {"content": '{"value": 1}'}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        },
    }
    result = processor.parse_results(json.dumps(raw))[0]
    assert result.success
    assert set(result.model_dump()) == {"custom_id", "result", "success"}
    assert not hasattr(result.result, "_raw_response")


def test_current_anthropic_beta_usage_is_not_recognized() -> None:
    beta = pytest.importorskip("anthropic.types.beta")
    response = beta.BetaMessage.model_validate(
        {
            "id": "audit",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "audit",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    assert not has_compatible_usage(response, _initialize_usage(Provider.ANTHROPIC))


def test_current_local_cache_replays_historical_raw_usage_without_total() -> None:
    from instructor.cache import AutoCache, load_cached_response, store_cached_response
    from instructor.v2.core.retry import _finalize_parsed_response

    class Item(BaseModel):
        value: int

    raw = completion(
        CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    original = _finalize_parsed_response(Item(value=1), raw, raw.usage)
    cache = AutoCache()
    store_cached_response(cache, "audit", original)
    restored = load_cached_response(cache, "audit", Item)
    assert restored._raw_response.usage.total_tokens == 15
    assert not isinstance(restored._raw_response, ChatCompletion)
    assert not hasattr(restored, "_total_usage")


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [(0, 0, 0), (None, 6, 6), (12, None, None), (True, 6, 6), (12, False, False)],
)
def test_anthropic_dictionary_thinking_preserves_non_counts(
    previous: int | None, current: int | None, expected: int | None
) -> None:
    types = pytest.importorskip("anthropic.types")
    if "output_tokens_details" in types.Usage.model_fields:
        pytest.skip(
            "Tests extra-dictionary behavior in SDKs without this declared field"
        )
    from instructor.v2.providers.anthropic.usage import initialize_usage
    from instructor.v2.providers.anthropic.usage import update_total_usage as accumulate

    total = initialize_usage()
    for count, unknown in [(previous, 100), (current, 200)]:
        usage = types.Usage.model_validate(
            {
                "input_tokens": 1,
                "output_tokens": 20,
                "output_tokens_details": {
                    "thinking_tokens": count,
                    "unknown_metadata": unknown,
                },
            }
        )
        assert accumulate(usage, total)
    details = total.model_dump()["output_tokens_details"]
    assert details["thinking_tokens"] == expected
    assert type(details["thinking_tokens"]) is type(expected)
    assert details["unknown_metadata"] == 200


def test_current_cli_mini_model_uses_full_model_price_table() -> None:
    from instructor.cli.usage import get_model_cost

    assert get_model_cost("gpt-4o-mini") is get_model_cost("gpt-4o")


def test_current_usage_event_copy_is_shared_between_handlers() -> None:
    from instructor.v2.core.hooks import Hooks
    from instructor.v2.core.retry import _usage_snapshot

    hooks = Hooks()
    observed: list[int] = []

    def change_event(usage: CompletionUsage) -> None:
        usage.total_tokens = 999

    def record_event(usage: CompletionUsage) -> None:
        observed.append(usage.total_tokens)

    hooks.on("completion:usage", change_event)
    hooks.on("completion:usage", record_event)
    total = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    hooks.emit_completion_usage(_usage_snapshot(total))
    assert total.total_tokens == 15
    assert observed == [999]

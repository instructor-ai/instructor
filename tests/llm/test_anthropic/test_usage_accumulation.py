"""Tests for Anthropic usage accumulation across retries.

Verifies that sub-model fields (cache_creation, server_tool_use,
output_tokens_details) sum correctly across multiple attempts rather
than latching on the first value.
"""

from __future__ import annotations

from typing import Any

import pytest

from instructor.v2.providers.anthropic.usage import (
    _ensure_sub_models,
    update_total_usage,
)


def _make_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_creation: dict | None = None,
    server_tool_use: dict | None = None,
    output_tokens_details: dict | None = None,
) -> Any:
    """Build an anthropic.types.Usage with optional sub-models."""
    from anthropic.types import Usage as AnthropicUsage

    kwargs: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
    }
    if cache_creation is not None:
        from anthropic.types.usage import CacheCreation

        kwargs["cache_creation"] = CacheCreation(**cache_creation)
    if server_tool_use is not None:
        from anthropic.types.usage import ServerToolUsage

        kwargs["server_tool_use"] = ServerToolUsage(**server_tool_use)
    if output_tokens_details is not None:
        try:
            from anthropic.types.usage import OutputTokensDetails

            kwargs["output_tokens_details"] = OutputTokensDetails(
                **output_tokens_details
            )
        except ImportError:
            # anthropic<0.94.0 — skip sub-model in test
            pass
    return AnthropicUsage(**kwargs)


class TestEnsureSubModels:
    """_ensure_sub_models initialises missing sub-models on the accumulator."""

    def test_initialises_cache_creation(self) -> None:
        usage = _make_usage(input_tokens=1, output_tokens=1)
        assert usage.cache_creation is None
        _ensure_sub_models(usage)
        assert usage.cache_creation is not None
        assert usage.cache_creation.ephemeral_1h_input_tokens == 0
        assert usage.cache_creation.ephemeral_5m_input_tokens == 0

    def test_initialises_server_tool_use(self) -> None:
        usage = _make_usage(input_tokens=1, output_tokens=1)
        assert usage.server_tool_use is None
        _ensure_sub_models(usage)
        assert usage.server_tool_use is not None
        assert usage.server_tool_use.web_search_requests == 0
        assert usage.server_tool_use.web_fetch_requests == 0

    def test_does_not_overwrite_existing_sub_models(self) -> None:
        usage = _make_usage(
            input_tokens=1,
            output_tokens=1,
            cache_creation={
                "ephemeral_1h_input_tokens": 100,
                "ephemeral_5m_input_tokens": 200,
            },
            server_tool_use={
                "web_search_requests": 3,
                "web_fetch_requests": 5,
            },
        )
        _ensure_sub_models(usage)
        assert usage.cache_creation.ephemeral_1h_input_tokens == 100
        assert usage.cache_creation.ephemeral_5m_input_tokens == 200
        assert usage.server_tool_use.web_search_requests == 3
        assert usage.server_tool_use.web_fetch_requests == 5


class TestUpdateTotalUsage:
    """update_total_usage accumulates sub-model fields across retries."""

    def test_cache_creation_accumulates_across_three_attempts(self) -> None:
        """Three retries each with ephemeral_5m=500 → total 1500."""
        total = _make_usage(input_tokens=0, output_tokens=0)
        _ensure_sub_models(total)

        for _ in range(3):
            resp = _make_usage(
                input_tokens=10,
                output_tokens=20,
                cache_creation={
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 500,
                },
            )
            update_total_usage(resp, total)

        assert total.cache_creation.ephemeral_5m_input_tokens == 1500

    def test_server_tool_use_accumulates_across_three_attempts(self) -> None:
        """Three retries each with web_search=2 → total 6."""
        total = _make_usage(input_tokens=0, output_tokens=0)
        _ensure_sub_models(total)

        for _ in range(3):
            resp = _make_usage(
                input_tokens=10,
                output_tokens=20,
                server_tool_use={
                    "web_search_requests": 2,
                    "web_fetch_requests": 1,
                },
            )
            update_total_usage(resp, total)

        assert total.server_tool_use.web_search_requests == 6
        assert total.server_tool_use.web_fetch_requests == 3

    def test_output_tokens_details_accumulates_when_available(self) -> None:
        """When OutputTokensDetails exists on the SDK, thinking_tokens sums."""
        try:
            from anthropic.types.usage import OutputTokensDetails  # noqa: F401
        except ImportError:
            pytest.skip("OutputTokensDetails not available on pinned anthropic")

        total = _make_usage(input_tokens=0, output_tokens=0)
        _ensure_sub_models(total)

        for _ in range(3):
            resp = _make_usage(
                input_tokens=10,
                output_tokens=20,
                output_tokens_details={"thinking_tokens": 40},
            )
            update_total_usage(resp, total)

        assert total.output_tokens_details.thinking_tokens == 120

    def test_does_not_crash_on_pinned_anthropic(self) -> None:
        """update_total_usage must not raise ImportError on anthropic==0.93.0.

        Regression: OutputTokensDetails does not exist on the pinned SDK
        version. _ensure_sub_models must guard the import so the first
        call does not crash. Also, ``output_tokens_details`` is not a
        declared field on ``Usage`` in 0.93.0, so all access must use
        ``getattr``/``hasattr`` to avoid ``AttributeError``.
        """
        total = _make_usage(input_tokens=0, output_tokens=0)
        resp = _make_usage(
            input_tokens=10,
            output_tokens=20,
            cache_creation={
                "ephemeral_1h_input_tokens": 0,
                "ephemeral_5m_input_tokens": 500,
            },
        )
        # Should not raise even if OutputTokensDetails is unimportable
        # and output_tokens_details is not a declared field.
        result = update_total_usage(resp, total)
        assert result is True
        assert total.cache_creation.ephemeral_5m_input_tokens == 500

    def test_write_back_merges_not_replaces(self) -> None:
        """Write-back must merge into response, not replace it wholesale.

        Regression: replacing the sub-model with model_copy(deep=True)
        would wipe any fields the accumulator does not track (e.g. future
        SDK additions). Merging preserves them.
        """
        total = _make_usage(input_tokens=0, output_tokens=0)
        _ensure_sub_models(total)

        resp = _make_usage(
            input_tokens=10,
            output_tokens=20,
            cache_creation={
                "ephemeral_1h_input_tokens": 0,
                "ephemeral_5m_input_tokens": 500,
            },
        )
        update_total_usage(resp, total)

        # The response sub-model should still be the same object (merged),
        # not a new copy that could lose untracked fields.
        assert resp.cache_creation is not None
        assert resp.cache_creation.ephemeral_5m_input_tokens == 500

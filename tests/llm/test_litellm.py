import os
import pytest
import instructor

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from litellm import acompletion, completion
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("litellm package is not installed", allow_module_level=True)


def test_litellm_create():
    client = instructor.from_litellm(completion)

    assert isinstance(client, instructor.Instructor)


def test_async_litellm_create():
    client = instructor.from_litellm(acompletion)

    assert isinstance(client, instructor.AsyncInstructor)


# ---------------------------------------------------------------------------
# Tests for update_total_usage preserving usage subclass type
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock
from openai.types.completion_usage import CompletionUsage as OpenAIUsage
from instructor.utils.core import update_total_usage


class _CustomUsage(OpenAIUsage):
    """A subclass of OpenAIUsage to simulate provider-specific usage types
    (e.g. litellm.types.utils.Usage)."""

    custom_field: str = "preserved"


def _make_response(usage):
    """Create a lightweight mock response with a .usage attribute."""
    resp = MagicMock()
    resp.usage = usage
    return resp


class TestUpdateTotalUsagePreservesType:
    """Ensure update_total_usage does NOT replace response.usage with
    total_usage, which would lose the original subclass type."""

    def test_usage_type_preserved_after_update(self):
        """response.usage should remain the same subclass instance."""
        response_usage = _CustomUsage(
            completion_tokens=10, prompt_tokens=20, total_tokens=30
        )
        total_usage = OpenAIUsage(
            completion_tokens=5, prompt_tokens=8, total_tokens=13
        )
        response = _make_response(response_usage)

        result = update_total_usage(response=response, total_usage=total_usage)

        # The type must still be the subclass, not plain OpenAIUsage
        assert type(result.usage) is _CustomUsage, (
            f"Expected _CustomUsage but got {type(result.usage).__name__}"
        )
        # The custom field should still be accessible
        assert result.usage.custom_field == "preserved"

    def test_usage_identity_preserved(self):
        """response.usage should be the exact same object (identity)."""
        response_usage = _CustomUsage(
            completion_tokens=10, prompt_tokens=20, total_tokens=30
        )
        total_usage = OpenAIUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )
        response = _make_response(response_usage)
        original_usage = response.usage

        update_total_usage(response=response, total_usage=total_usage)

        assert response.usage is original_usage

    def test_totals_accumulated_on_response_usage(self):
        """response.usage tokens should reflect the running total."""
        response_usage = _CustomUsage(
            completion_tokens=10, prompt_tokens=20, total_tokens=30
        )
        total_usage = OpenAIUsage(
            completion_tokens=5, prompt_tokens=8, total_tokens=13
        )
        response = _make_response(response_usage)

        update_total_usage(response=response, total_usage=total_usage)

        # total_usage should have accumulated
        assert total_usage.completion_tokens == 15
        assert total_usage.prompt_tokens == 28
        assert total_usage.total_tokens == 43

        # response.usage should mirror the totals
        assert response.usage.completion_tokens == 15
        assert response.usage.prompt_tokens == 28
        assert response.usage.total_tokens == 43

    def test_totals_accumulated_across_multiple_calls(self):
        """Simulates multiple retries accumulating into total_usage while
        each response keeps its own subclass type."""
        total_usage = OpenAIUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )

        # First call
        r1_usage = _CustomUsage(
            completion_tokens=10, prompt_tokens=20, total_tokens=30
        )
        r1 = _make_response(r1_usage)
        update_total_usage(response=r1, total_usage=total_usage)

        assert type(r1.usage) is _CustomUsage
        assert r1.usage.completion_tokens == 10
        assert total_usage.completion_tokens == 10

        # Second call
        r2_usage = _CustomUsage(
            completion_tokens=5, prompt_tokens=8, total_tokens=13
        )
        r2 = _make_response(r2_usage)
        update_total_usage(response=r2, total_usage=total_usage)

        assert type(r2.usage) is _CustomUsage
        assert r2.usage.completion_tokens == 15  # accumulated
        assert r2.usage.prompt_tokens == 28
        assert r2.usage.total_tokens == 43
        assert total_usage.completion_tokens == 15

    def test_none_response_returns_none(self):
        """Passing None as response should return None without error."""
        total_usage = OpenAIUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )
        result = update_total_usage(response=None, total_usage=total_usage)
        assert result is None

    def test_plain_openai_usage_still_works(self):
        """A plain OpenAIUsage (not a subclass) should still work correctly."""
        response_usage = OpenAIUsage(
            completion_tokens=10, prompt_tokens=20, total_tokens=30
        )
        total_usage = OpenAIUsage(
            completion_tokens=5, prompt_tokens=8, total_tokens=13
        )
        response = _make_response(response_usage)

        update_total_usage(response=response, total_usage=total_usage)

        assert total_usage.completion_tokens == 15
        assert response.usage.completion_tokens == 15
        assert type(response.usage) is OpenAIUsage

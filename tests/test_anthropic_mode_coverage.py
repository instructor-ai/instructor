"""Tests for complete Anthropic mode coverage in initialize_usage and streaming.

Regression tests for:
- initialize_usage returning correct Usage type for all Anthropic modes
- Streaming handlers recognizing ANTHROPIC_REASONING_TOOLS mode
"""

import pytest

from instructor.mode import Mode
from instructor.core.retry import initialize_usage


class TestInitializeUsage:
    """Verify initialize_usage returns Anthropic Usage for all Anthropic modes."""

    @pytest.mark.parametrize(
        "mode",
        [
            Mode.ANTHROPIC_TOOLS,
            Mode.ANTHROPIC_JSON,
            Mode.ANTHROPIC_REASONING_TOOLS,
            Mode.ANTHROPIC_PARALLEL_TOOLS,
        ],
        ids=[
            "ANTHROPIC_TOOLS",
            "ANTHROPIC_JSON",
            "ANTHROPIC_REASONING_TOOLS",
            "ANTHROPIC_PARALLEL_TOOLS",
        ],
    )
    def test_anthropic_modes_return_anthropic_usage(self, mode: Mode) -> None:
        """All Anthropic modes must return Anthropic Usage, not OpenAI CompletionUsage."""
        from anthropic.types import Usage as AnthropicUsage

        usage = initialize_usage(mode)
        assert isinstance(usage, AnthropicUsage), (
            f"Mode {mode.value} returned {type(usage).__name__} instead of AnthropicUsage"
        )
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    @pytest.mark.parametrize(
        "mode",
        [
            Mode.TOOLS,
            Mode.JSON,
            Mode.GEMINI_TOOLS,
            Mode.GEMINI_JSON,
        ],
        ids=["TOOLS", "JSON", "GEMINI_TOOLS", "GEMINI_JSON"],
    )
    def test_non_anthropic_modes_return_openai_usage(self, mode: Mode) -> None:
        """Non-Anthropic modes must return OpenAI CompletionUsage."""
        from openai.types import CompletionUsage

        usage = initialize_usage(mode)
        assert isinstance(usage, CompletionUsage), (
            f"Mode {mode.value} returned {type(usage).__name__} instead of CompletionUsage"
        )
        assert usage.completion_tokens == 0
        assert usage.prompt_tokens == 0

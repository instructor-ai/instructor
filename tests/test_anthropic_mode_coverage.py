"""Tests for complete Anthropic mode coverage in initialize_usage, streaming, and multimodal.

Regression tests for:
- initialize_usage returning correct Usage type for all Anthropic modes
- Streaming handlers recognizing ANTHROPIC_REASONING_TOOLS mode
- Multimodal content conversion using to_anthropic() for all Anthropic modes
- from_response dispatching to parse_anthropic_tools for ANTHROPIC_REASONING_TOOLS
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestStreamingAnthropicModeCoverage:
    """Verify streaming code recognizes ANTHROPIC_REASONING_TOOLS."""

    @pytest.mark.parametrize(
        "mode",
        [Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS],
        ids=["ANTHROPIC_TOOLS", "ANTHROPIC_REASONING_TOOLS"],
    )
    def test_partial_extract_json_yields_partial_json(self, mode: Mode) -> None:
        """Both ANTHROPIC_TOOLS and ANTHROPIC_REASONING_TOOLS must yield chunk.delta.partial_json."""
        from instructor.dsl.partial import PartialBase

        chunk = SimpleNamespace(delta=SimpleNamespace(partial_json='{"name": "test"}'))
        chunks = iter([chunk])

        partial_cls = PartialBase
        # Call the extract_json generator directly
        gen = partial_cls.extract_json(chunks, mode)
        results = list(gen)

        assert len(results) == 1
        assert results[0] == '{"name": "test"}'

    @pytest.mark.parametrize(
        "mode",
        [Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS],
        ids=["ANTHROPIC_TOOLS", "ANTHROPIC_REASONING_TOOLS"],
    )
    def test_iterable_extract_json_yields_partial_json(self, mode: Mode) -> None:
        """Both ANTHROPIC_TOOLS and ANTHROPIC_REASONING_TOOLS must yield chunk.delta.partial_json in IterableBase."""
        from instructor.dsl.iterable import IterableBase

        chunk = SimpleNamespace(delta=SimpleNamespace(partial_json='{"item": 1}'))
        chunks = iter([chunk])

        gen = IterableBase.extract_json(chunks, mode)
        results = list(gen)

        assert len(results) == 1
        assert results[0] == '{"item": 1}'


class TestMultimodalAnthropicModeCoverage:
    """Verify multimodal content conversion uses to_anthropic() for all Anthropic modes."""

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
    def test_image_content_uses_to_anthropic(self, mode: Mode) -> None:
        """Image content must call to_anthropic() for all Anthropic modes, not to_openai()."""
        from instructor.processing.multimodal import Image, convert_contents

        img = Image(
            source="data:image/png;base64,iVBOR",
            media_type="image/png",
            data="iVBOR",
        )

        with (
            patch.object(
                Image,
                "to_anthropic",
                return_value={"type": "image", "source": {"type": "base64"}},
            ) as mock_to_anthropic,
            patch.object(Image, "to_openai") as mock_to_openai,
        ):
            result = convert_contents([img], mode)

        mock_to_anthropic.assert_called_once()
        mock_to_openai.assert_not_called()
        assert result[0]["type"] == "image"


class TestFromResponseAnthropicModeCoverage:
    """Verify from_response dispatches correctly for ANTHROPIC_REASONING_TOOLS."""

    def test_reasoning_tools_calls_parse_anthropic_tools(self) -> None:
        """ANTHROPIC_REASONING_TOOLS must dispatch to parse_anthropic_tools, not fall through."""
        from instructor.processing.function_calls import OpenAISchema

        mock_completion = MagicMock()
        mock_ctx = {"test": True}

        with patch.object(
            OpenAISchema,
            "parse_anthropic_tools",
            return_value=MagicMock(),
        ) as mock_parse:
            OpenAISchema.from_response(
                mock_completion,
                mode=Mode.ANTHROPIC_REASONING_TOOLS,
                validation_context=mock_ctx,
            )
            mock_parse.assert_called_once_with(mock_completion, mock_ctx, None)

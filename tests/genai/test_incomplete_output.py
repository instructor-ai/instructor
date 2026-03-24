"""Tests for IncompleteOutputException raised on MAX_TOKENS in genai responses."""

import pytest
from unittest.mock import MagicMock
import instructor
from instructor.core.exceptions import IncompleteOutputException


class _SimpleModel(instructor.OpenAISchema):  # type: ignore[misc]
    name: str


def _make_completion(finish_reason) -> MagicMock:
    """Build a minimal mock GenerateContentResponse with the given finish_reason."""
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    completion = MagicMock()
    completion.candidates = [candidate]
    return completion


def test_parse_genai_structured_outputs_raises_on_max_tokens() -> None:
    """parse_genai_structured_outputs should raise IncompleteOutputException when MAX_TOKENS."""
    from google.genai import types

    completion = _make_completion(types.FinishReason.MAX_TOKENS)

    with pytest.raises(IncompleteOutputException):
        _SimpleModel.parse_genai_structured_outputs(completion)  # type: ignore[attr-defined]


def test_parse_genai_structured_outputs_does_not_raise_on_stop() -> None:
    """parse_genai_structured_outputs should not raise when finish_reason is STOP."""
    from google.genai import types

    completion = _make_completion(types.FinishReason.STOP)
    completion.text = '{"name": "Alice"}'

    result = _SimpleModel.parse_genai_structured_outputs(completion)  # type: ignore[attr-defined]
    assert result.name == "Alice"

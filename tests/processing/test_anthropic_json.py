"""Isolated tests for Anthropic JSON parsing helpers."""

from anthropic.types import Message, Usage
import pytest
from pydantic import ValidationError

import instructor


CONTROL_CHAR_JSON = """{
"data": "Claude likes
control
characters"
}"""


class _AnthropicTestModel(instructor.OpenAISchema):  # type: ignore[misc]
    data: str


def _build_message(data_content: str) -> Message:
    return Message(
        id="test_id",
        content=[{"type": "text", "text": data_content}],
        model="claude-3-haiku-20240307",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def test_parse_anthropic_json_strict_control_characters() -> None:
    message = _build_message(CONTROL_CHAR_JSON)

    with pytest.raises(ValidationError):
        _AnthropicTestModel.parse_anthropic_json(message, strict=True)  # type: ignore[arg-type]


def test_parse_anthropic_json_non_strict_preserves_control_characters() -> None:
    message = _build_message(CONTROL_CHAR_JSON)

    model = _AnthropicTestModel.parse_anthropic_json(message, strict=False)  # type: ignore[arg-type]

    assert model.data == "Claude likes\ncontrol\ncharacters"

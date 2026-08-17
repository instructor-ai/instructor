"""Regression tests for merging the JSON schema instruction into a system message.

JSON/MD_JSON handlers append their generated schema instruction to the leading
system message. When that message carries structured content (a list of parts)
the handlers used to write to `content[0]["text"]` unconditionally, which raised
`KeyError: 'text'` for a system message that leads with a non-text part (an
image, for example) and `IndexError` for an empty content list.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.messages import merge_system_instruction
from instructor.v2.providers.mistral.handlers import MistralMDJSONHandler
from instructor.v2.providers.openai.handlers import (
    OpenAIJSONHandler,
    OpenAIMDJSONHandler,
)
from instructor.v2.providers.writer.handlers import WriterMDJSONHandler
from instructor.v2.providers.xai.handlers import XAIMDJSONHandler

INSTRUCTION = "return json"


class Answer(BaseModel):
    name: str


def test_merge_system_instruction_appends_to_string_content() -> None:
    messages = merge_system_instruction(
        [{"role": "system", "content": "be terse"}], INSTRUCTION
    )

    assert messages == [{"role": "system", "content": f"be terse\n\n{INSTRUCTION}"}]


def test_merge_system_instruction_appends_to_first_text_part() -> None:
    messages = merge_system_instruction(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "be terse"},
                    {"type": "text", "text": "and polite"},
                ],
            }
        ],
        INSTRUCTION,
    )

    assert messages[0]["content"] == [
        {"type": "text", "text": f"be terse\n\n{INSTRUCTION}"},
        {"type": "text", "text": "and polite"},
    ]


def test_merge_system_instruction_skips_leading_non_text_part() -> None:
    image_part = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    messages = merge_system_instruction(
        [
            {
                "role": "system",
                "content": [image_part, {"type": "text", "text": "be terse"}],
            }
        ],
        INSTRUCTION,
    )

    assert messages[0]["content"] == [
        image_part,
        {"type": "text", "text": f"be terse\n\n{INSTRUCTION}"},
    ]


@pytest.mark.parametrize(
    "content",
    [[], [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]],
    ids=["empty", "image-only"],
)
def test_merge_system_instruction_adds_text_part_when_none_exists(
    content: list[dict[str, Any]],
) -> None:
    messages = merge_system_instruction(
        [{"role": "system", "content": list(content)}], INSTRUCTION
    )

    assert messages[0]["content"] == [*content, {"type": "text", "text": INSTRUCTION}]


def test_merge_system_instruction_prepends_when_no_system_message() -> None:
    user_message = {"role": "user", "content": "hi"}

    messages = merge_system_instruction([user_message], INSTRUCTION)

    assert messages == [{"role": "system", "content": INSTRUCTION}, user_message]


def test_merge_system_instruction_prepends_for_unusable_content() -> None:
    system_message = {"role": "system", "content": None}

    messages = merge_system_instruction([system_message], INSTRUCTION)

    assert messages == [{"role": "system", "content": INSTRUCTION}, system_message]


@pytest.mark.parametrize(
    "handler_cls",
    [
        OpenAIJSONHandler,
        OpenAIMDJSONHandler,
        MistralMDJSONHandler,
        WriterMDJSONHandler,
        XAIMDJSONHandler,
    ],
    ids=lambda handler_cls: handler_cls.__name__,
)
def test_prepare_request_handles_system_message_without_leading_text(
    handler_cls: type[ModeHandler],
) -> None:
    image_part = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    kwargs: dict[str, Any] = {
        "model": "test",
        "messages": [
            {"role": "system", "content": [image_part, {"type": "text", "text": "hi"}]},
            {"role": "user", "content": "Ada"},
        ],
    }

    _, new_kwargs = handler_cls().prepare_request(Answer, kwargs)

    system_content = new_kwargs["messages"][0]["content"]
    assert image_part in system_content
    text_parts = [
        part["text"]
        for part in system_content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert any("json_schema" in text for text in text_parts)

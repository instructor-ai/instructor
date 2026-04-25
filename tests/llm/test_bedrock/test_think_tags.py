"""Tests for think-tag stripping in Bedrock response parsing.

Bedrock reasoning models (e.g. Kimi K2 Thinking) prefix their actual response
with a <think>...</think> block.  These tests verify that instructor strips those
tags before attempting JSON extraction so that both BEDROCK_JSON and MD_JSON
modes work without spurious retries.
"""
from __future__ import annotations

from pydantic import BaseModel

from instructor.processing.function_calls import openai_schema


class _User(BaseModel):
    name: str
    age: int


User = openai_schema(_User)


# ---------------------------------------------------------------------------
# Helper: build a minimal Bedrock converse() response dict
# ---------------------------------------------------------------------------

def _bedrock_response(text: str) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
    }


# ---------------------------------------------------------------------------
# parse_bedrock_json — BEDROCK_JSON mode
# ---------------------------------------------------------------------------

class TestParseBedrockJsonThinkTags:
    def test_plain_json_no_think_tags(self):
        response = _bedrock_response('{"name": "Jason", "age": 10}')
        user = User.parse_bedrock_json(response)
        assert user.name == "Jason"
        assert user.age == 10

    def test_think_tags_before_json(self):
        text = "<think>Let me figure out the answer...</think>\n{\"name\": \"Jason\", \"age\": 10}"
        response = _bedrock_response(text)
        user = User.parse_bedrock_json(response)
        assert user.name == "Jason"
        assert user.age == 10

    def test_think_tags_with_braces_inside(self):
        """Think block may contain braces that would confuse naive JSON extraction."""
        text = (
            "<think>I'll return {\"key\": \"value\"} as the structure.</think>\n"
            '{"name": "Alice", "age": 30}'
        )
        response = _bedrock_response(text)
        user = User.parse_bedrock_json(response)
        assert user.name == "Alice"
        assert user.age == 30

    def test_think_tags_with_json_in_codeblock(self):
        text = (
            "<think>Reasoning goes here.</think>\n"
            "```json\n{\"name\": \"Bob\", \"age\": 25}\n```"
        )
        response = _bedrock_response(text)
        user = User.parse_bedrock_json(response)
        assert user.name == "Bob"
        assert user.age == 25

    def test_multiline_think_block(self):
        text = (
            "<think>\n"
            "Line one of reasoning.\n"
            "Line two with {json-like} content.\n"
            "</think>\n"
            '{"name": "Carol", "age": 22}'
        )
        response = _bedrock_response(text)
        user = User.parse_bedrock_json(response)
        assert user.name == "Carol"
        assert user.age == 22


# ---------------------------------------------------------------------------
# _extract_text_content — used by parse_json / MD_JSON path
# ---------------------------------------------------------------------------

class TestExtractTextContentThinkTags:
    """_extract_text_content strips think tags so parse_json (MD_JSON) works."""

    def test_strips_think_tags_from_bedrock_dict(self):
        from instructor.processing.function_calls import _extract_text_content

        text = "<think>Thinking...</think>\n{\"name\": \"Dave\", \"age\": 40}"
        response = _bedrock_response(text)
        extracted = _extract_text_content(response)
        assert "<think>" not in extracted
        assert "Dave" in extracted

    def test_no_think_tags_passes_through(self):
        from instructor.processing.function_calls import _extract_text_content

        text = '{"name": "Eve", "age": 35}'
        response = _bedrock_response(text)
        extracted = _extract_text_content(response)
        assert extracted == text


# ---------------------------------------------------------------------------
# reask_md_json — retry path for MD_JSON with Bedrock dict response
# ---------------------------------------------------------------------------

class TestReaskMdJsonBedrockDict:
    def test_reask_extends_messages_with_bedrock_dict(self):
        from instructor.providers.openai.utils import reask_md_json

        response = _bedrock_response('{"bad": "json"}')
        kwargs = {
            "messages": [{"role": "user", "content": [{"text": "Tell me about Jason"}]}],
        }
        exception = ValueError("name field required")

        initial_len = len(kwargs["messages"])
        new_kwargs = reask_md_json(kwargs, response, exception)

        # Three messages: original + assistant (from response) + user correction
        assert len(new_kwargs["messages"]) == initial_len + 2
        # Last message is a Bedrock-format user correction
        last = new_kwargs["messages"][-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert "name field required" in last["content"][0]["text"]

    def test_reask_with_openai_response_unchanged(self):
        """Non-Bedrock responses still use the existing choices-based path."""
        from unittest.mock import MagicMock
        from instructor.providers.openai.utils import reask_md_json

        message = MagicMock()
        message.role = "assistant"
        message.content = '{"bad": "json"}'
        message.tool_calls = None
        message.function_call = None

        response = MagicMock()
        response.choices = [MagicMock(message=message)]

        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        exception = ValueError("age required")
        initial_len = len(kwargs["messages"])

        new_kwargs = reask_md_json(kwargs, response, exception)
        # Should not crash and should extend messages by at least 1
        assert len(new_kwargs["messages"]) > initial_len

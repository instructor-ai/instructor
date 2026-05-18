"""Unit tests for Bedrock v2 handlers."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    """Simple answer model for tests."""

    answer: float


class User(BaseModel):
    """User model for tests."""

    name: str
    age: int


def _bedrock_tool_response(
    args: dict[str, Any], tool_use_id: str = "tool-use-1", name: str = "Answer"
) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": name,
                            "input": args,
                        }
                    }
                ]
            }
        }
    }


def _bedrock_text_response(text: str) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "text": text,
                    }
                ]
            }
        }
    }


class TestBedrockToolsHandler:
    """Tests for BedrockToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        return mode_registry.get_handlers(Provider.BEDROCK, Mode.TOOLS)

    def test_prepare_request_with_none_model(self, handler):
        """prepare_request returns unchanged model and converted kwargs."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_tool_config(self, handler):
        """prepare_request adds Bedrock tool config."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        assert "toolConfig" in result_kwargs
        assert "tools" in result_kwargs["toolConfig"]
        assert "toolChoice" in result_kwargs["toolConfig"]

    def test_parse_response_from_tool_use(self, handler):
        """parse_response extracts tool input."""
        response = _bedrock_tool_response({"answer": 4.0})
        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_handle_reask_adds_messages(self, handler):
        """handle_reask adds tool error messages."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = _bedrock_tool_response({"answer": "bad"})
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert "messages" in result
        assert len(result["messages"]) > 1


class TestBedrockMDJSONHandler:
    """Tests for BedrockMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        return mode_registry.get_handlers(Provider.BEDROCK, Mode.MD_JSON)

    def test_prepare_request_with_none_model(self, handler):
        """prepare_request returns unchanged model and converted kwargs."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_system_message(self, handler):
        """prepare_request adds system instructions for JSON output."""
        kwargs = {"messages": [{"role": "user", "content": "Extract user"}]}
        result_model, result_kwargs = handler.request_handler(User, kwargs)

        assert result_model is not None
        assert "system" in result_kwargs

    def test_parse_response_from_text(self, handler):
        """parse_response extracts JSON from Bedrock text."""
        response = _bedrock_text_response('{"answer": 4}')
        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_parse_response_from_codeblock(self, handler):
        """parse_response handles JSON in code blocks."""
        response = _bedrock_text_response('```json\n{"answer": 3}\n```')
        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 3.0

    def test_handle_reask_adds_messages(self, handler):
        """handle_reask adds user correction message."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = _bedrock_text_response("Invalid response")
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert "messages" in result
        assert len(result["messages"]) > 1

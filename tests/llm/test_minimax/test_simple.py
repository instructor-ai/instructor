"""Unit tests for MiniMax provider."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import openai

import instructor
from instructor.providers.minimax.client import from_minimax
from instructor.providers.minimax.utils import (
    handle_minimax_tools,
    handle_minimax_json,
    reask_minimax_tools,
    reask_minimax_json,
)


class TestFromMiniMax:
    """Tests for from_minimax client factory."""

    def test_creates_sync_instructor(self):
        client = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        result = from_minimax(client, mode=instructor.Mode.MINIMAX_TOOLS)
        assert isinstance(result, instructor.Instructor)
        assert result.provider == instructor.Provider.MINIMAX
        assert result.mode == instructor.Mode.MINIMAX_TOOLS

    def test_creates_async_instructor(self):
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        result = from_minimax(client, mode=instructor.Mode.MINIMAX_TOOLS)
        assert isinstance(result, instructor.AsyncInstructor)
        assert result.provider == instructor.Provider.MINIMAX
        assert result.mode == instructor.Mode.MINIMAX_TOOLS

    def test_default_mode_is_tools(self):
        client = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        result = from_minimax(client)
        assert result.mode == instructor.Mode.MINIMAX_TOOLS

    def test_json_mode(self):
        client = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        result = from_minimax(client, mode=instructor.Mode.MINIMAX_JSON)
        assert result.mode == instructor.Mode.MINIMAX_JSON

    def test_rejects_invalid_mode(self):
        client = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        with pytest.raises(Exception):
            from_minimax(client, mode=instructor.Mode.TOOLS)

    def test_rejects_invalid_client(self):
        with pytest.raises(Exception):
            from_minimax("not-a-client")  # type: ignore


class TestHandleMiniMaxTools:
    """Tests for MiniMax tools mode handler."""

    def test_adds_tools_and_tool_choice(self):
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        kwargs: dict = {"messages": [{"role": "user", "content": "test"}]}
        _, result = handle_minimax_tools(User, kwargs)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert "tool_choice" in result
        assert result["tool_choice"]["type"] == "function"


class TestHandleMiniMaxJSON:
    """Tests for MiniMax JSON mode handler."""

    def test_prepends_system_instruction(self):
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        kwargs: dict = {"messages": [{"role": "user", "content": "test"}]}
        _, result = handle_minimax_json(User, kwargs)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "JSON" in result["messages"][0]["content"]
        assert "User" in result["messages"][0]["content"]

    def test_does_not_add_response_format(self):
        from pydantic import BaseModel

        class User(BaseModel):
            name: str

        kwargs: dict = {"messages": [{"role": "user", "content": "test"}]}
        _, result = handle_minimax_json(User, kwargs)

        assert "response_format" not in result


class TestReaskMiniMaxTools:
    """Tests for MiniMax tools reask handler."""

    def test_reask_with_stream_response(self):
        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_tools(kwargs, None, ValueError("test error"))

        assert len(result["messages"]) == 2
        assert "Validation Error" in result["messages"][-1]["content"]

    def test_reask_with_valid_response(self):
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "User"
        mock_tool_call.function.arguments = '{"name": "test"}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_tools(kwargs, mock_response, ValueError("test error"))

        assert len(result["messages"]) > 1


class TestReaskMiniMaxJSON:
    """Tests for MiniMax JSON reask handler."""

    def test_reask_appends_correction_message(self):
        mock_message = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_json(kwargs, mock_response, ValueError("bad json"))

        assert len(result["messages"]) > 1
        assert "Correct your JSON" in result["messages"][-1]["content"]


class TestMiniMaxProviderEnum:
    """Tests for MiniMax in Provider enum."""

    def test_minimax_in_provider_enum(self):
        assert hasattr(instructor.Provider, "MINIMAX")
        assert instructor.Provider.MINIMAX.value == "minimax"


class TestMiniMaxModeEnum:
    """Tests for MiniMax modes in Mode enum."""

    def test_minimax_tools_mode(self):
        assert hasattr(instructor.Mode, "MINIMAX_TOOLS")
        assert instructor.Mode.MINIMAX_TOOLS.value == "minimax_tools"

    def test_minimax_json_mode(self):
        assert hasattr(instructor.Mode, "MINIMAX_JSON")
        assert instructor.Mode.MINIMAX_JSON.value == "minimax_json"

    def test_minimax_tools_in_tool_modes(self):
        assert instructor.Mode.MINIMAX_TOOLS in instructor.Mode.tool_modes()

    def test_minimax_json_in_json_modes(self):
        assert instructor.Mode.MINIMAX_JSON in instructor.Mode.json_modes()

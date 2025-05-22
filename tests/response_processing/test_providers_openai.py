"""Tests for OpenAI-specific handlers."""

import pytest
from unittest.mock import patch

from pydantic import BaseModel

from instructor.exceptions import ConfigurationError
from instructor.mode import Mode
from instructor.response_processing.providers.openai import (
    O1JSONHandler,
    OpenAIFunctionsHandler,
    OpenAIJSONHandler,
    OpenAIParallelToolsHandler,
    OpenAIToolsHandler,
    OpenAIToolsStrictHandler,
    OpenRouterStructuredOutputsHandler,
    ResponsesToolsHandler,
)


class MockModel(BaseModel):
    """Mock model for testing."""

    name: str
    value: int

    @property
    def openai_schema(self):
        return {
            "name": "MockModel",
            "description": "A mock model",
            "parameters": self.model_json_schema(),
        }


class TestOpenAIToolsHandler:
    """Test OpenAI tools handler."""

    def test_handle(self):
        """Test handling OpenAI tools mode."""
        handler = OpenAIToolsHandler()
        kwargs = {"test": "value"}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert "tools" in result_kwargs
        assert len(result_kwargs["tools"]) == 1
        assert result_kwargs["tools"][0]["type"] == "function"
        assert result_kwargs["tool_choice"]["type"] == "function"
        assert result_kwargs["tool_choice"]["function"]["name"] == "MockModel"


class TestOpenAIToolsStrictHandler:
    """Test OpenAI strict tools handler."""

    @patch("instructor.response_processing.providers.openai.pydantic_function_tool")
    def test_handle(self, mock_pydantic_tool):
        """Test handling OpenAI strict tools mode."""
        mock_pydantic_tool.return_value = {
            "function": {"name": "MockModel", "parameters": {}, "strict": False}
        }

        handler = OpenAIToolsStrictHandler()
        kwargs = {"test": "value"}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert "tools" in result_kwargs
        assert result_kwargs["tools"][0]["function"]["strict"] is True
        mock_pydantic_tool.assert_called_once_with(MockModel)


class TestOpenAIFunctionsHandler:
    """Test OpenAI functions handler (deprecated)."""

    def test_handle_warns_deprecation(self):
        """Test handling warns about deprecation."""
        handler = OpenAIFunctionsHandler()
        kwargs = {"test": "value"}

        with patch.object(Mode, "warn_mode_functions_deprecation") as mock_warn:
            response_model, result_kwargs = handler.handle(MockModel, kwargs)
            mock_warn.assert_called_once()

        assert response_model is MockModel
        assert "functions" in result_kwargs
        assert "function_call" in result_kwargs


class TestOpenAIJSONHandler:
    """Test OpenAI JSON handler."""

    def test_handle_json_mode(self):
        """Test handling JSON mode."""
        handler = OpenAIJSONHandler(Mode.JSON)
        kwargs = {"messages": []}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert result_kwargs["response_format"] == {"type": "json_object"}
        assert len(result_kwargs["messages"]) == 1
        assert result_kwargs["messages"][0]["role"] == "system"

    def test_handle_json_schema_mode(self):
        """Test handling JSON schema mode."""
        handler = OpenAIJSONHandler(Mode.JSON_SCHEMA)
        kwargs = {"messages": []}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert result_kwargs["response_format"]["type"] == "json_object"
        assert "schema" in result_kwargs["response_format"]

    def test_handle_md_json_mode(self):
        """Test handling markdown JSON mode."""
        handler = OpenAIJSONHandler(Mode.MD_JSON)
        kwargs = {"messages": [{"role": "user", "content": "test"}]}

        with patch(
            "instructor.response_processing.providers.openai.merge_consecutive_messages"
        ) as mock_merge:
            mock_merge.return_value = kwargs["messages"]
            response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert any(
            "```json" in msg["content"]
            for msg in result_kwargs["messages"]
            if msg["role"] == "user"
        )


class TestOpenAIParallelToolsHandler:
    """Test OpenAI parallel tools handler."""

    def test_handle_success(self):
        """Test successful handling."""
        from collections.abc import Iterable

        handler = OpenAIParallelToolsHandler()
        kwargs = {"stream": False}

        with patch(
            "instructor.response_processing.providers.openai.handle_parallel_model"
        ) as mock_handle:
            mock_handle.return_value = [{"tool": "definition"}]
            response_model, result_kwargs = handler.handle(Iterable[MockModel], kwargs)

        # ParallelModel returns an instance, check its type
        assert response_model.__class__.__name__ == "ParallelBase"
        assert result_kwargs["tools"] == [{"tool": "definition"}]
        assert result_kwargs["tool_choice"] == "auto"

    def test_handle_stream_error(self):
        """Test error when streaming is enabled."""
        handler = OpenAIParallelToolsHandler()
        kwargs = {"stream": True}

        with pytest.raises(ConfigurationError, match="stream=True is not supported"):
            handler.handle(MockModel, kwargs)


class TestO1JSONHandler:
    """Test O1 JSON handler."""

    def test_handle_success(self):
        """Test successful handling."""
        handler = O1JSONHandler()
        kwargs = {"messages": [{"role": "user", "content": "test"}]}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert len(result_kwargs["messages"]) == 2
        assert "json_schema" in result_kwargs["messages"][-1]["content"]

    def test_handle_system_message_error(self):
        """Test error with system messages."""
        handler = O1JSONHandler()
        kwargs = {"messages": [{"role": "system", "content": "system"}]}

        with pytest.raises(ValueError, match="System messages are not supported"):
            handler.handle(MockModel, kwargs)


class TestResponsesToolsHandler:
    """Test responses tools handler."""

    @patch("instructor.response_processing.providers.openai.pydantic_function_tool")
    def test_handle_without_inbuilt_tools(self, mock_pydantic_tool):
        """Test handling without inbuilt tools."""
        mock_pydantic_tool.return_value = {
            "function": {"name": "MockModel", "parameters": {}, "strict": True}
        }

        handler = ResponsesToolsHandler(with_inbuilt_tools=False)
        kwargs = {"max_tokens": 100}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert "tools" in result_kwargs
        assert "max_output_tokens" in result_kwargs
        assert "max_tokens" not in result_kwargs

    @patch("instructor.response_processing.providers.openai.pydantic_function_tool")
    def test_handle_with_inbuilt_tools_empty(self, mock_pydantic_tool):
        """Test handling with inbuilt tools when no tools exist."""
        mock_pydantic_tool.return_value = {
            "function": {
                "name": "MockModel",
                "parameters": {},
                "description": "Test description",
                "strict": True,
            }
        }

        handler = ResponsesToolsHandler(with_inbuilt_tools=True)
        kwargs = {}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert "tools" in result_kwargs
        assert len(result_kwargs["tools"]) == 1
        assert "tool_choice" in result_kwargs

    @patch("instructor.response_processing.providers.openai.pydantic_function_tool")
    def test_handle_with_inbuilt_tools_existing(self, mock_pydantic_tool):
        """Test handling with inbuilt tools when tools already exist."""
        mock_pydantic_tool.return_value = {
            "function": {"name": "MockModel", "parameters": {}, "strict": True}
        }

        handler = ResponsesToolsHandler(with_inbuilt_tools=True)
        kwargs = {"tools": [{"existing": "tool"}]}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert len(result_kwargs["tools"]) == 2
        assert result_kwargs["tools"][0] == {"existing": "tool"}


class TestOpenRouterStructuredOutputsHandler:
    """Test OpenRouter structured outputs handler."""

    def test_handle(self):
        """Test handling OpenRouter structured outputs."""
        handler = OpenRouterStructuredOutputsHandler()
        kwargs = {}

        response_model, result_kwargs = handler.handle(MockModel, kwargs)

        assert response_model is MockModel
        assert "response_format" in result_kwargs
        assert result_kwargs["response_format"]["type"] == "json_schema"
        assert result_kwargs["response_format"]["json_schema"]["name"] == "MockModel"
        assert result_kwargs["response_format"]["json_schema"]["strict"] is True
        assert (
            result_kwargs["response_format"]["json_schema"]["schema"][
                "additionalProperties"
            ]
            is False
        )

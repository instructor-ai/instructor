"""Tests for MiniMax provider integration."""

import os
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

import pytest
import instructor
from instructor.providers.minimax.client import from_minimax
from instructor.providers.minimax.utils import (
    handle_minimax_tools,
    handle_minimax_json,
    reask_minimax_tools,
    reask_minimax_json,
)
from pydantic import BaseModel


# ============================================================
# Test Models
# ============================================================


class User(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    country: str


def _make_tool_response_mock(tool_name="User", arguments='{"name": "test"}'):
    """Create a properly mocked response for tool calling tests."""
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = tool_name
    mock_tool_call.function.arguments = arguments

    mock_message = MagicMock()
    mock_message.role = "assistant"
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call]
    mock_message.function_call = None
    mock_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def _make_json_response_mock(content='{"name": "test", "age": 25}'):
    """Create a properly mocked response for JSON mode tests."""
    mock_message = MagicMock()
    mock_message.role = "assistant"
    mock_message.content = content
    mock_message.tool_calls = None
    mock_message.function_call = None
    mock_message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
    }

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


# ============================================================
# Unit Tests - Mode Handlers
# ============================================================


class TestHandleMiniMaxTools:
    """Unit tests for handle_minimax_tools."""

    def test_adds_tools_and_tool_choice(self):
        new_kwargs: dict = {"messages": [{"role": "user", "content": "test"}]}
        result_model, result_kwargs = handle_minimax_tools(User, new_kwargs)

        assert result_model is User
        assert "tools" in result_kwargs
        assert len(result_kwargs["tools"]) == 1
        assert result_kwargs["tools"][0]["type"] == "function"
        assert "tool_choice" in result_kwargs
        assert result_kwargs["tool_choice"]["type"] == "function"

    def test_preserves_existing_kwargs(self):
        new_kwargs: dict = {
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
        }
        _, result_kwargs = handle_minimax_tools(User, new_kwargs)

        assert result_kwargs["temperature"] == 0.7
        assert "tools" in result_kwargs

    def test_schema_has_correct_properties(self):
        new_kwargs: dict = {"messages": []}
        _, result_kwargs = handle_minimax_tools(User, new_kwargs)

        schema = result_kwargs["tools"][0]["function"]
        assert "parameters" in schema
        props = schema["parameters"]["properties"]
        assert "name" in props
        assert "age" in props

    def test_with_complex_model(self):
        new_kwargs: dict = {"messages": []}
        _, result_kwargs = handle_minimax_tools(Address, new_kwargs)

        schema = result_kwargs["tools"][0]["function"]
        props = schema["parameters"]["properties"]
        assert "street" in props
        assert "city" in props
        assert "country" in props


class TestHandleMiniMaxJSON:
    """Unit tests for handle_minimax_json."""

    def test_prepends_system_instruction(self):
        new_kwargs: dict = {"messages": [{"role": "user", "content": "test"}]}
        result_model, result_kwargs = handle_minimax_json(User, new_kwargs)

        assert result_model is User
        assert len(result_kwargs["messages"]) == 2
        assert result_kwargs["messages"][0]["role"] == "system"
        assert "JSON" in result_kwargs["messages"][0]["content"]
        assert "User" in result_kwargs["messages"][0]["content"]

    def test_system_instruction_contains_schema(self):
        new_kwargs: dict = {"messages": []}
        _, result_kwargs = handle_minimax_json(User, new_kwargs)

        system_msg = result_kwargs["messages"][0]["content"]
        assert "name" in system_msg
        assert "age" in system_msg
        assert "model_validate_json" in system_msg

    def test_preserves_existing_kwargs(self):
        new_kwargs: dict = {
            "messages": [],
            "model": "MiniMax-M2.7",
        }
        _, result_kwargs = handle_minimax_json(User, new_kwargs)

        assert result_kwargs["model"] == "MiniMax-M2.7"
        assert result_kwargs["messages"][0]["role"] == "system"


class TestReaskMiniMaxTools:
    """Unit tests for reask_minimax_tools."""

    def test_extends_messages_with_error(self):
        mock_response = _make_tool_response_mock()

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_tools(
            kwargs, mock_response, ValueError("age is required")
        )

        assert len(result["messages"]) > 1
        last_msg = result["messages"][-1]
        assert "Validation Error" in last_msg["content"]
        assert "age is required" in last_msg["content"]

    def test_does_not_mutate_original_kwargs(self):
        mock_response = _make_tool_response_mock()

        kwargs = {"messages": [{"role": "user", "content": "original"}]}
        result = reask_minimax_tools(kwargs, mock_response, ValueError("err"))

        assert kwargs is not result

    def test_multiple_tool_calls(self):
        mock_tool_call1 = MagicMock()
        mock_tool_call1.function.name = "User"
        mock_tool_call1.function.arguments = '{"name": "test"}'
        mock_tool_call2 = MagicMock()
        mock_tool_call2.function.name = "Address"
        mock_tool_call2.function.arguments = '{"street": "test"}'

        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]
        mock_message.function_call = None
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_tools(kwargs, mock_response, ValueError("err"))

        error_msgs = [m for m in result["messages"] if "Validation Error" in m.get("content", "")]
        assert len(error_msgs) == 2


class TestReaskMiniMaxJSON:
    """Unit tests for reask_minimax_json."""

    def test_extends_messages_with_correction_request(self):
        mock_response = _make_json_response_mock()

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_json(
            kwargs, mock_response, ValueError("invalid JSON")
        )

        assert len(result["messages"]) > 1
        last_msg = result["messages"][-1]
        assert "Correct your JSON" in last_msg["content"]
        assert "invalid JSON" in last_msg["content"]

    def test_does_not_mutate_original_kwargs(self):
        mock_response = _make_json_response_mock()

        kwargs = {"messages": [{"role": "user", "content": "original"}]}
        result = reask_minimax_json(kwargs, mock_response, ValueError("err"))

        assert kwargs is not result

    def test_includes_assistant_message(self):
        mock_response = _make_json_response_mock(content='{"incomplete": true}')

        kwargs = {"messages": [{"role": "user", "content": "test"}]}
        result = reask_minimax_json(kwargs, mock_response, ValueError("missing fields"))

        roles = [m.get("role", "") for m in result["messages"]]
        assert "assistant" in roles
        assert "user" in roles


# ============================================================
# Unit Tests - Client Creation
# ============================================================


class TestFromMiniMax:
    """Unit tests for from_minimax client creation."""

    def test_sync_client_creation(self):
        import openai

        mock_client = MagicMock(spec=openai.OpenAI)
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = MagicMock()

        result = from_minimax(mock_client, mode=instructor.Mode.MINIMAX_TOOLS)
        assert isinstance(result, instructor.Instructor)

    def test_async_client_creation(self):
        import openai

        mock_client = MagicMock(spec=openai.AsyncOpenAI)
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        result = from_minimax(mock_client, mode=instructor.Mode.MINIMAX_TOOLS)
        assert isinstance(result, instructor.AsyncInstructor)

    def test_invalid_mode_raises_error(self):
        import openai

        mock_client = MagicMock(spec=openai.OpenAI)

        with pytest.raises(Exception):
            from_minimax(mock_client, mode=instructor.Mode.TOOLS)

    def test_invalid_client_raises_error(self):
        with pytest.raises(Exception):
            from_minimax("not_a_client", mode=instructor.Mode.MINIMAX_TOOLS)

    def test_json_mode(self):
        import openai

        mock_client = MagicMock(spec=openai.OpenAI)
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = MagicMock()

        result = from_minimax(mock_client, mode=instructor.Mode.MINIMAX_JSON)
        assert isinstance(result, instructor.Instructor)

    def test_default_mode_is_tools(self):
        import openai

        mock_client = MagicMock(spec=openai.OpenAI)
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = MagicMock()

        result = from_minimax(mock_client)
        assert isinstance(result, instructor.Instructor)


# ============================================================
# Unit Tests - Provider and Mode Registration
# ============================================================


class TestProviderRegistration:
    """Unit tests for MiniMax provider registration."""

    def test_minimax_in_provider_enum(self):
        assert hasattr(instructor.Provider, "MINIMAX")
        assert instructor.Provider.MINIMAX.value == "minimax"

    def test_minimax_tools_mode_exists(self):
        assert hasattr(instructor.Mode, "MINIMAX_TOOLS")

    def test_minimax_json_mode_exists(self):
        assert hasattr(instructor.Mode, "MINIMAX_JSON")

    def test_minimax_tools_in_tool_modes(self):
        assert instructor.Mode.MINIMAX_TOOLS in instructor.Mode.tool_modes()

    def test_minimax_json_in_json_modes(self):
        assert instructor.Mode.MINIMAX_JSON in instructor.Mode.json_modes()

    def test_from_minimax_importable(self):
        from instructor import from_minimax
        assert callable(from_minimax)

    def test_minimax_in_supported_providers(self):
        from instructor.auto_client import supported_providers
        assert "minimax" in supported_providers


class TestProviderDetection:
    """Unit tests for MiniMax provider URL detection."""

    def test_detect_minimax_from_url(self):
        from instructor.utils.providers import get_provider, Provider
        assert get_provider("https://api.minimax.io/v1") == Provider.MINIMAX

    def test_detect_minimax_from_url_partial(self):
        from instructor.utils.providers import get_provider, Provider
        assert get_provider("https://minimax.example.com") == Provider.MINIMAX


# ============================================================
# Integration Tests (require MINIMAX_API_KEY)
# ============================================================


@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxIntegration:
    """Integration tests that call the MiniMax API."""

    def test_from_provider_sync(self):
        """Test sync client via from_provider."""
        client = instructor.from_provider(
            "minimax/MiniMax-M2.7",
            api_key=os.getenv("MINIMAX_API_KEY"),
        )

        user = client.create(
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            response_model=User,
        )

        assert isinstance(user, User)
        assert user.name == "Jason"
        assert user.age == 25

    def test_from_minimax_tools_mode(self):
        """Test direct from_minimax with TOOLS mode."""
        import openai

        openai_client = openai.OpenAI(
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url="https://api.minimax.io/v1",
        )
        client = from_minimax(openai_client, mode=instructor.Mode.MINIMAX_TOOLS)

        user = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "user", "content": "Extract: Alice is 30 years old"},
            ],
            response_model=User,
        )

        assert isinstance(user, User)
        assert user.name == "Alice"
        assert user.age == 30

    def test_from_minimax_json_mode(self):
        """Test direct from_minimax with JSON mode."""
        import openai

        openai_client = openai.OpenAI(
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url="https://api.minimax.io/v1",
        )
        client = from_minimax(openai_client, mode=instructor.Mode.MINIMAX_JSON)

        user = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "user", "content": "Extract: Bob is 42 years old"},
            ],
            response_model=User,
        )

        assert isinstance(user, User)
        assert user.name == "Bob"
        assert user.age == 42

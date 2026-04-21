"""Unit tests for the MiniMax provider.

These tests exercise the ``from_minimax`` factory and the mode dispatch paths
without making real API calls. MiniMax exposes an OpenAI-compatible API, so
the tests use mocked ``openai`` clients to assert that Instructor configures
requests correctly for each supported mode.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import openai
import pytest
from pydantic import BaseModel

import instructor
from instructor.providers.minimax.client import (
    DEFAULT_MINIMAX_BASE_URL,
    from_minimax,
)
from instructor.providers.minimax.utils import (
    MINIMAX_HANDLERS,
    handle_minimax_json,
    handle_minimax_tools,
    reask_minimax_json,
    reask_minimax_tools,
)


class User(BaseModel):
    name: str
    age: int


def _make_sync_client() -> openai.OpenAI:
    return openai.OpenAI(api_key="test-key", base_url=DEFAULT_MINIMAX_BASE_URL)


def _make_async_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key="test-key", base_url=DEFAULT_MINIMAX_BASE_URL)


def test_from_minimax_returns_sync_instructor():
    client = _make_sync_client()
    result = from_minimax(client)

    assert isinstance(result, instructor.Instructor)
    assert result.provider == instructor.Provider.MINIMAX
    assert result.mode == instructor.Mode.MINIMAX_TOOLS


def test_from_minimax_returns_async_instructor():
    client = _make_async_client()
    result = from_minimax(client, mode=instructor.Mode.MINIMAX_JSON)

    assert isinstance(result, instructor.AsyncInstructor)
    assert result.provider == instructor.Provider.MINIMAX
    assert result.mode == instructor.Mode.MINIMAX_JSON


def test_from_minimax_builds_default_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    result = from_minimax()

    assert isinstance(result, instructor.Instructor)
    assert isinstance(result.client, openai.OpenAI)
    assert str(result.client.base_url).rstrip("/") == DEFAULT_MINIMAX_BASE_URL


def test_from_minimax_builds_async_default_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    result = from_minimax(async_client=True, mode=instructor.Mode.MINIMAX_JSON)

    assert isinstance(result, instructor.AsyncInstructor)
    assert isinstance(result.client, openai.AsyncOpenAI)


def test_from_minimax_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch):
    from instructor.core.exceptions import ConfigurationError

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="MINIMAX_API_KEY"):
        from_minimax()


def test_from_minimax_invalid_mode_raises():
    from instructor.core.exceptions import ModeError

    client = _make_sync_client()

    with pytest.raises(ModeError):
        from_minimax(client, mode=instructor.Mode.TOOLS)


def test_from_minimax_invalid_client_raises():
    from instructor.core.exceptions import ClientError

    with pytest.raises(ClientError):
        from_minimax(object())  # type: ignore[arg-type]


def test_handle_minimax_tools_adds_tool_definition():
    response_model, kwargs = handle_minimax_tools(
        User, {"messages": [{"role": "user", "content": "Extract"}]}
    )

    assert response_model is User
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "User"},
    }
    assert len(kwargs["tools"]) == 1
    tool = kwargs["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "User"
    assert "parameters" in tool["function"]


def test_handle_minimax_tools_noop_without_response_model():
    response_model, kwargs = handle_minimax_tools(None, {"messages": []})

    assert response_model is None
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_handle_minimax_json_adds_response_format_and_schema_message():
    response_model, kwargs = handle_minimax_json(
        User, {"messages": [{"role": "user", "content": "Extract"}]}
    )

    assert response_model is User
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"][-1]["role"] == "user"
    # The appended message must include the JSON schema for the model.
    schema_fragment = json.dumps(
        User.model_json_schema(), indent=2, ensure_ascii=False
    )
    assert schema_fragment in kwargs["messages"][-1]["content"]


def test_handle_minimax_json_noop_without_response_model():
    response_model, kwargs = handle_minimax_json(None, {"messages": []})

    assert response_model is None
    assert "response_format" not in kwargs


def _make_tool_response() -> MagicMock:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    tool_call = ChatCompletionMessageToolCall(
        id="call-1",
        type="function",
        function=Function(name="User", arguments="{}"),
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def _make_json_response() -> MagicMock:
    from openai.types.chat import ChatCompletionMessage

    message = ChatCompletionMessage(role="assistant", content="{}")
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def test_reask_minimax_tools_appends_tool_messages():
    response = _make_tool_response()

    new_kwargs = reask_minimax_tools(
        {"messages": [{"role": "user", "content": "Extract"}]},
        response,
        ValueError("bad"),
    )

    roles = [m["role"] for m in new_kwargs["messages"]]
    assert "tool" in roles
    tool_msg = next(m for m in new_kwargs["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call-1"
    assert tool_msg["name"] == "User"
    assert "Validation Error" in tool_msg["content"]


def test_reask_minimax_json_appends_user_correction():
    response = _make_json_response()

    new_kwargs = reask_minimax_json(
        {"messages": [{"role": "user", "content": "Extract"}]},
        response,
        ValueError("bad"),
    )

    last = new_kwargs["messages"][-1]
    assert last["role"] == "user"
    assert "JSON ONLY RESPONSE" in last["content"]


def test_minimax_handlers_registry():
    assert set(MINIMAX_HANDLERS.keys()) == {
        instructor.Mode.MINIMAX_TOOLS,
        instructor.Mode.MINIMAX_JSON,
    }
    for entry in MINIMAX_HANDLERS.values():
        assert callable(entry["response"])
        assert callable(entry["reask"])


def test_from_provider_minimax_uses_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    client = instructor.from_provider("minimax/MiniMax-Text-01")

    assert isinstance(client, instructor.Instructor)
    assert client.provider == instructor.Provider.MINIMAX
    assert client.mode == instructor.Mode.MINIMAX_TOOLS


def test_from_provider_minimax_missing_key(monkeypatch: pytest.MonkeyPatch):
    from instructor.core.exceptions import ConfigurationError

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="MINIMAX_API_KEY"):
        instructor.from_provider("minimax/MiniMax-Text-01")


def test_get_provider_detects_minimax_base_url():
    from instructor.utils.providers import get_provider

    assert get_provider("https://api.minimax.chat/v1") == instructor.Provider.MINIMAX


def test_mode_classifications_include_minimax():
    assert instructor.Mode.MINIMAX_TOOLS in instructor.Mode.tool_modes()
    assert instructor.Mode.MINIMAX_JSON in instructor.Mode.json_modes()

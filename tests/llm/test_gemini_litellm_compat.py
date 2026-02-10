"""Tests for GEMINI_JSON and GEMINI_TOOLS mode compatibility with LiteLLM.

These tests verify that Gemini modes work when the model is passed in kwargs
(the LiteLLM pattern) rather than being bound at client creation time
(the native Gemini SDK pattern).

Regression tests for https://github.com/jxnl/instructor/issues/1489
"""

from __future__ import annotations

import json
from unittest.mock import patch

from pydantic import BaseModel, Field

from instructor.providers.gemini.utils import handle_gemini_json, handle_gemini_tools


class PersonalInfo(BaseModel):
    name: str = Field(..., description="The person's name.")
    age: int = Field(..., description="The person's age.")


# --- GEMINI_JSON with LiteLLM (model in kwargs) ---


def test_gemini_json_litellm_path_injects_schema():
    """When model is in kwargs, handle_gemini_json should inject JSON schema into system message."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "John Doe is 30 years old."},
        ],
    }

    response_model, result = handle_gemini_json(PersonalInfo, kwargs)

    assert response_model is PersonalInfo
    # Schema should be injected into a new system message
    assert result["messages"][0]["role"] == "system"
    assert "json_schema" in result["messages"][0]["content"]
    assert "PersonalInfo" in json.dumps(result["messages"][0]["content"])


def test_gemini_json_litellm_path_sets_response_format():
    """LiteLLM path should set response_format, not generation_config."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "John Doe is 30 years old."},
        ],
    }

    _, result = handle_gemini_json(PersonalInfo, kwargs)

    assert result["response_format"] == {"type": "json_object"}
    assert "generation_config" not in result


def test_gemini_json_litellm_path_preserves_model():
    """Model should remain in kwargs for LiteLLM routing."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "test"},
        ],
    }

    _, result = handle_gemini_json(PersonalInfo, kwargs)

    assert result["model"] == "gemini/gemini-2.0-flash"


def test_gemini_json_litellm_path_keeps_openai_message_format():
    """LiteLLM path should NOT convert messages to Gemini contents format."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "John Doe is 30 years old."},
        ],
    }

    _, result = handle_gemini_json(PersonalInfo, kwargs)

    # Messages should still be in OpenAI format (not converted to "contents")
    assert "messages" in result
    assert "contents" not in result


def test_gemini_json_litellm_path_no_response_model():
    """LiteLLM path with no response_model should be a no-op."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }

    response_model, result = handle_gemini_json(None, kwargs)

    assert response_model is None
    assert result["model"] == "gemini/gemini-2.0-flash"
    assert "response_format" not in result


def test_gemini_json_litellm_path_appends_to_existing_system():
    """If a system message already exists, schema should be appended."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "John Doe is 30 years old."},
        ],
    }

    _, result = handle_gemini_json(PersonalInfo, kwargs)

    assert result["messages"][0]["role"] == "system"
    assert "You are a helpful assistant." in result["messages"][0]["content"]
    assert "json_schema" in result["messages"][0]["content"]


# --- GEMINI_JSON with native Gemini SDK (no model in kwargs) ---


def test_gemini_json_native_path_no_model():
    """Native Gemini path (no model in kwargs) should use generation_config and update_gemini_kwargs."""
    kwargs = {
        "messages": [
            {"role": "user", "content": "John Doe is 30 years old."},
        ],
    }

    with patch(
        "instructor.providers.gemini.utils.update_gemini_kwargs",
        side_effect=lambda k: k,
    ) as mock_update:
        _, result = handle_gemini_json(PersonalInfo, kwargs)
        mock_update.assert_called_once()

    assert (
        result.get("generation_config", {}).get("response_mime_type")
        == "application/json"
    )
    assert "response_format" not in result


# --- GEMINI_TOOLS with LiteLLM (model in kwargs) ---


def test_gemini_tools_litellm_path_uses_openai_schema():
    """When model is in kwargs, handle_gemini_tools should use OpenAI-style tool definitions."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "John Doe is 30 years old."},
        ],
    }

    response_model, result = handle_gemini_tools(PersonalInfo, kwargs)

    assert response_model is PersonalInfo
    # Should use OpenAI-style tools format
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "function"
    assert "function" in result["tools"][0]
    assert result["tools"][0]["function"]["name"] == "PersonalInfo"


def test_gemini_tools_litellm_path_sets_tool_choice():
    """LiteLLM path should set OpenAI-style tool_choice, not Gemini tool_config."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "test"},
        ],
    }

    _, result = handle_gemini_tools(PersonalInfo, kwargs)

    assert "tool_choice" in result
    assert result["tool_choice"]["type"] == "function"
    assert result["tool_choice"]["function"]["name"] == "PersonalInfo"
    assert "tool_config" not in result


def test_gemini_tools_litellm_path_preserves_model():
    """Model should remain in kwargs for LiteLLM routing."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "test"},
        ],
    }

    _, result = handle_gemini_tools(PersonalInfo, kwargs)

    assert result["model"] == "gemini/gemini-2.0-flash"


def test_gemini_tools_litellm_path_keeps_openai_message_format():
    """LiteLLM path should NOT convert messages to Gemini contents format."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "test"},
        ],
    }

    _, result = handle_gemini_tools(PersonalInfo, kwargs)

    assert "messages" in result
    assert "contents" not in result


def test_gemini_tools_litellm_path_no_response_model():
    """LiteLLM path with no response_model should be a no-op."""
    kwargs = {
        "model": "gemini/gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }

    response_model, result = handle_gemini_tools(None, kwargs)

    assert response_model is None
    assert result["model"] == "gemini/gemini-2.0-flash"
    assert "tools" not in result


# --- Regression test matching exact reproduction from issue #1489 ---


def test_gemini_json_litellm_router_reproduction():
    """
    Regression test for #1489: Mode.GEMINI_JSON with LiteLLM Router.

    Previously raised:
        ConfigurationError: Gemini `model` must be set while patching the client,
        not passed as a parameter to the create method
    """
    kwargs = {
        "model": "gemini-2.0-flash",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in extracting data in structured JSON formats.",
            },
            {"role": "user", "content": "John Doe is 87 years old."},
        ],
    }

    # This should NOT raise ConfigurationError anymore
    response_model, result = handle_gemini_json(PersonalInfo, kwargs)

    assert response_model is PersonalInfo
    assert result["model"] == "gemini-2.0-flash"
    assert result["response_format"] == {"type": "json_object"}
    assert "messages" in result
    assert "contents" not in result

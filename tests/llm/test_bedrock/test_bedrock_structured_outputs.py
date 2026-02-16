"""Unit tests for Bedrock BEDROCK_STRUCTURED_OUTPUTS and BEDROCK_TOOLS_STRICT modes.

These are pure unit tests (no real API calls) exercising the v2 Bedrock handlers
and the registry wiring that backs the two native constrained-decoding modes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry, normalize_mode
from instructor.v2.core.mode import reset_deprecated_mode_warnings
from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.providers.bedrock.handlers import (
    handle_bedrock_structured_outputs,
    handle_bedrock_tools_strict,
    reask_bedrock_structured_outputs,
    reask_bedrock_tools_strict,
)


# --- Test models ---


class SimpleUser(BaseModel):
    """A simple user model for testing."""

    name: str
    age: int


class NestedAddress(BaseModel):
    """A nested address model."""

    street: str
    city: str
    zip_code: str = Field(description="5-digit ZIP code")


class UserWithAddress(BaseModel):
    """A user with a nested address."""

    name: str
    age: int
    address: NestedAddress


# --- handler tests: BEDROCK_STRUCTURED_OUTPUTS ---


class TestHandleBedrockStructuredOutputs:
    def test_returns_none_when_no_response_model(self):
        result_model, result_kwargs = handle_bedrock_structured_outputs(
            None,
            {"model": "anthropic.claude-sonnet-4-5-v1", "messages": []},
        )
        assert result_model is None
        assert "outputConfig" not in result_kwargs

    def test_sets_output_config(self):
        _, kwargs = handle_bedrock_structured_outputs(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "Extract user info"}],
            },
        )
        assert "outputConfig" in kwargs
        text_format = kwargs["outputConfig"]["textFormat"]
        assert text_format["type"] == "json_schema"
        json_schema = text_format["structure"]["jsonSchema"]
        assert json_schema["name"] == "SimpleUser"
        # schema must be a JSON string, not a dict
        assert isinstance(json_schema["schema"], str)
        parsed_schema = json.loads(json_schema["schema"])
        assert "properties" in parsed_schema
        assert "name" in parsed_schema["properties"]
        assert "age" in parsed_schema["properties"]
        # Bedrock requires additionalProperties: false on object schemas
        assert parsed_schema["additionalProperties"] is False

    def test_does_not_set_tool_config(self):
        _, kwargs = handle_bedrock_structured_outputs(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert "toolConfig" not in kwargs

    def test_normalizes_kwargs(self):
        _, kwargs = handle_bedrock_structured_outputs(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "temperature": 0.5,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        # model -> modelId
        assert "modelId" in kwargs
        assert "model" not in kwargs
        # temperature/max_tokens -> inferenceConfig
        assert kwargs["inferenceConfig"]["temperature"] == 0.5
        assert kwargs["inferenceConfig"]["maxTokens"] == 100

    def test_nested_model_schema(self):
        _, kwargs = handle_bedrock_structured_outputs(
            UserWithAddress,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        schema_str = kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"][
            "schema"
        ]
        parsed = json.loads(schema_str)
        # Should reference the nested model
        assert "address" in parsed["properties"]
        # additionalProperties: false must be applied to nested $defs too
        nested = parsed["$defs"]["NestedAddress"]
        assert nested["additionalProperties"] is False

    def test_uses_docstring_as_description(self):
        _, kwargs = handle_bedrock_structured_outputs(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        desc = kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"][
            "description"
        ]
        assert desc == "A simple user model for testing."


# --- handler tests: BEDROCK_TOOLS_STRICT ---


class TestHandleBedrockToolsStrict:
    def test_returns_none_when_no_response_model(self):
        result_model, result_kwargs = handle_bedrock_tools_strict(
            None,
            {"model": "anthropic.claude-sonnet-4-5-v1", "messages": []},
        )
        assert result_model is None
        assert "toolConfig" not in result_kwargs

    def test_sets_strict_on_tool_spec(self):
        _, kwargs = handle_bedrock_tools_strict(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert "toolConfig" in kwargs
        tool_spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
        assert tool_spec["strict"] is True
        assert tool_spec["name"] == "SimpleUser"
        # additionalProperties: false must be applied to the tool input schema
        assert tool_spec["inputSchema"]["json"]["additionalProperties"] is False

    def test_tool_choice_set(self):
        _, kwargs = handle_bedrock_tools_strict(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "SimpleUser"}}

    def test_does_not_set_output_config(self):
        _, kwargs = handle_bedrock_tools_strict(
            SimpleUser,
            {
                "model": "anthropic.claude-sonnet-4-5-v1",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert "outputConfig" not in kwargs


# --- reask tests ---


class TestReaskBedrockStructuredOutputs:
    def test_appends_messages(self):
        original_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": [{"text": "Extract user"}]},
            ],
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {"jsonSchema": {"schema": "{}", "name": "Test"}},
                }
            },
        }
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": '{"name": "John"}'}],
                }
            }
        }
        exception = ValueError("Missing field: age")

        result = reask_bedrock_structured_outputs(original_kwargs, response, exception)

        # Should have 3 messages now: original user + assistant + correction user
        assert len(result["messages"]) == 3
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["role"] == "user"
        assert "age" in result["messages"][2]["content"][0]["text"]
        # outputConfig should be preserved
        assert "outputConfig" in result


class TestReaskBedrockToolsStrict:
    def test_delegates_to_reask_bedrock_tools(self):
        original_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": [{"text": "test"}]},
            ],
        }
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool_123",
                                "name": "SimpleUser",
                                "input": {"name": "John"},
                            }
                        }
                    ],
                }
            }
        }
        exception = ValueError("Missing field: age")
        result = reask_bedrock_tools_strict(original_kwargs, response, exception)

        # Should have appended assistant + tool result error messages
        assert len(result["messages"]) == 3
        tool_result = result["messages"][2]["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "tool_123"
        assert tool_result["status"] == "error"


# --- registry wiring tests ---


class TestBedrockRegistry:
    def test_structured_outputs_alias_normalizes_to_json_schema(self):
        # The provider-named alias is a deprecated shim for canonical JSON_SCHEMA:
        # it still resolves correctly but warns to steer users to the canonical mode.
        reset_deprecated_mode_warnings()
        with pytest.warns(DeprecationWarning):
            result = normalize_mode(Provider.BEDROCK, Mode.BEDROCK_STRUCTURED_OUTPUTS)
        assert result is Mode.JSON_SCHEMA

    def test_tools_strict_alias_normalizes_to_tools_strict(self):
        reset_deprecated_mode_warnings()
        with pytest.warns(DeprecationWarning):
            result = normalize_mode(Provider.BEDROCK, Mode.BEDROCK_TOOLS_STRICT)
        assert result is Mode.TOOLS_STRICT

    def test_structured_outputs_handler_registered(self):
        assert mode_registry.is_registered(
            Provider.BEDROCK, Mode.BEDROCK_STRUCTURED_OUTPUTS
        )
        handlers = mode_registry.get_handlers(
            Provider.BEDROCK, Mode.BEDROCK_STRUCTURED_OUTPUTS
        )
        assert handlers.response_parser is not None
        assert handlers.reask_handler is not None

    def test_tools_strict_handler_registered(self):
        assert mode_registry.is_registered(Provider.BEDROCK, Mode.BEDROCK_TOOLS_STRICT)
        handlers = mode_registry.get_handlers(
            Provider.BEDROCK, Mode.BEDROCK_TOOLS_STRICT
        )
        assert handlers.response_parser is not None
        assert handlers.reask_handler is not None


# --- parse tests (via registered handlers) ---


class TestParseBedrockStructuredOutputs:
    @pytest.fixture
    def parser(self):
        return mode_registry.get_handlers(
            Provider.BEDROCK, Mode.BEDROCK_STRUCTURED_OUTPUTS
        ).response_parser

    def test_parse_dict_response(self, parser):
        completion = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": '{"name": "Alice", "age": 30}'}],
                }
            }
        }
        result = parser(completion, SimpleUser)
        assert isinstance(result, SimpleUser)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_missing_text_raises(self, parser):
        completion = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"image": {}}],
                }
            }
        }
        with pytest.raises(ResponseParsingError):
            parser(completion, SimpleUser)


class TestParseBedrockToolsStrict:
    @pytest.fixture
    def parser(self):
        return mode_registry.get_handlers(
            Provider.BEDROCK, Mode.BEDROCK_TOOLS_STRICT
        ).response_parser

    def test_parse_tool_use_response(self, parser):
        """BEDROCK_TOOLS_STRICT parses tool use the same way as BEDROCK_TOOLS."""
        completion = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "SimpleUser",
                                "input": {"name": "Bob", "age": 25},
                            }
                        }
                    ],
                }
            }
        }
        result = parser(completion, SimpleUser)
        assert isinstance(result, SimpleUser)
        assert result.name == "Bob"
        assert result.age == 25


# --- Mode classification tests ---


class TestModeClassification:
    def test_structured_outputs_is_json_mode(self):
        assert Mode.BEDROCK_STRUCTURED_OUTPUTS in Mode.json_modes()

    def test_structured_outputs_is_not_tool_mode(self):
        assert Mode.BEDROCK_STRUCTURED_OUTPUTS not in Mode.tool_modes()

    def test_tools_strict_is_tool_mode(self):
        assert Mode.BEDROCK_TOOLS_STRICT in Mode.tool_modes()

    def test_tools_strict_is_not_json_mode(self):
        assert Mode.BEDROCK_TOOLS_STRICT not in Mode.json_modes()

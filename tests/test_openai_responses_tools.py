from unittest.mock import MagicMock

from pydantic import BaseModel

from openai import pydantic_function_tool

from instructor.v2.providers.openai.handlers import (
    OpenAIResponsesToolsHandler,
    reask_responses_tools,
)


def _tool_parameters_schema(model):
    """Get the parameters schema as pydantic_function_tool produces it."""
    return pydantic_function_tool(model)["function"]["parameters"]


class ResponseToolModel(BaseModel):
    """Extract a structured response for the user."""

    name: str


class AlternateModel(BaseModel):
    """A different schema used to test conflict detection."""

    title: str
    count: int


# ── prepare_request ─────────────────────────────────────────────────────


def test_responses_tools_preserves_function_description() -> None:
    expected_description = pydantic_function_tool(ResponseToolModel)["function"][
        "description"
    ]

    handler = OpenAIResponsesToolsHandler()
    _, kwargs = handler.prepare_request(ResponseToolModel, {})

    assert kwargs["tools"][0]["description"] == expected_description


def test_responses_tools_sets_text_format() -> None:
    """prepare_request must set text.format = json_schema from the
    response model so both output paths are aligned."""

    handler = OpenAIResponsesToolsHandler()
    _, kwargs = handler.prepare_request(ResponseToolModel, {})

    text = kwargs.get("text")
    assert text is not None, "text config should be set"
    fmt = text["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "ResponseToolModel"
    assert fmt["strict"] is True
    assert fmt["schema"] == _tool_parameters_schema(ResponseToolModel)
    # Must include additionalProperties: false for OpenAI strict mode
    assert fmt["schema"].get("additionalProperties") is False


def test_responses_tools_overrides_conflicting_text_format() -> None:
    """If the user provides a text.format with a different schema, it must be
    overridden to match the tool schema."""

    conflicting_text = {
        "format": {
            "type": "json_schema",
            "name": "AlternateModel",
            "strict": True,
            "schema": AlternateModel.model_json_schema(),
        }
    }

    handler = OpenAIResponsesToolsHandler()
    _, kwargs = handler.prepare_request(
        ResponseToolModel, {"text": conflicting_text}
    )

    fmt = kwargs["text"]["format"]
    assert fmt["name"] == "ResponseToolModel"
    assert fmt["schema"] == _tool_parameters_schema(ResponseToolModel)


def test_responses_tools_preserves_matching_text_format() -> None:
    """If the user provides a text.format that already matches the tool
    schema, it should be left unchanged (no override)."""

    matching_text = {
        "format": {
            "type": "json_schema",
            "name": "ResponseToolModel",
            "strict": True,
            "schema": _tool_parameters_schema(ResponseToolModel),
        }
    }

    handler = OpenAIResponsesToolsHandler()
    _, kwargs = handler.prepare_request(
        ResponseToolModel, {"text": matching_text}
    )

    # Should be the exact same object (not replaced)
    assert kwargs["text"] is matching_text


def test_responses_tools_none_model_no_text() -> None:
    """When response_model is None, text config should not be set."""

    handler = OpenAIResponsesToolsHandler()
    _, kwargs = handler.prepare_request(None, {})
    assert "text" not in kwargs


# ── reask_responses_tools ───────────────────────────────────────────────


def _make_mock_response(arguments: str) -> MagicMock:
    """Create a mock response with a single tool call."""
    tool_call = MagicMock()
    tool_call.type = "function_call"
    tool_call.arguments = arguments
    tool_call.name = "ResponseToolModel"
    tool_call.id = "call_123"

    response = MagicMock()
    response.output = [tool_call]
    return response


def test_reask_responses_tools_empty_args_message() -> None:
    """When tool call returns empty '{}', the retry message should explicitly
    tell the model to populate all required fields."""

    response = _make_mock_response("{}")
    kwargs = {"messages": []}
    error = ValueError("1 validation error for ResponseToolModel\nname\n  Field required")

    result = reask_responses_tools(kwargs, response, error)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]["content"]

    # Must contain targeted guidance, not the generic "fix the errors with {}"
    assert "empty arguments" in msg
    assert "MUST populate ALL required fields" in msg
    assert "fix the errors with" not in msg


def test_reask_responses_tools_nonempty_args_message() -> None:
    """When tool call has non-empty but invalid args, the retry message should
    use the standard format with the actual arguments."""

    response = _make_mock_response('{"name": 123}')
    kwargs = {"messages": []}
    error = ValueError("1 validation error for ResponseToolModel\nname\n  Input should be a valid string")

    result = reask_responses_tools(kwargs, response, error)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]["content"]

    # Should use the standard retry format
    assert "fix the errors with" in msg
    assert '{"name": 123}' in msg


def test_reask_responses_tools_none_arguments() -> None:
    """When tool call has arguments=None, treat as empty."""

    tool_call = MagicMock()
    tool_call.type = "function_call"
    tool_call.arguments = None
    tool_call.name = "ResponseToolModel"
    tool_call.id = "call_123"

    response = MagicMock()
    response.output = [tool_call]
    kwargs = {"messages": []}
    error = ValueError("Field required")

    result = reask_responses_tools(kwargs, response, error)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]["content"]
    assert "MUST populate ALL required fields" in msg


def test_responses_tools_overrides_text_type_format() -> None:
    """If the user provides text.format with type 'text', it should be
    overridden with json_schema to prevent competing output paths."""

    text_format = {"format": {"type": "text"}}

    handler = OpenAIResponsesToolsHandler()
    _, kwargs = handler.prepare_request(
        ResponseToolModel, {"text": text_format}
    )

    fmt = kwargs["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "ResponseToolModel"


def test_parse_response_warns_on_empty_args(caplog) -> None:
    """parse_response should log a warning when tool args are empty."""
    import logging

    tool_call = MagicMock()
    tool_call.type = "function_call"
    tool_call.arguments = "{}"
    tool_call.name = "ResponseToolModel"

    response = MagicMock()
    response.output = [tool_call]
    # Remove choices so the fallback also fails cleanly
    response.choices = []

    handler = OpenAIResponsesToolsHandler()
    with caplog.at_level(logging.WARNING, logger="instructor"):
        try:
            handler.parse_response(response, ResponseToolModel)
        except Exception:
            pass  # Expected — validation or fallback will fail

    assert any("empty arguments" in r.message for r in caplog.records)


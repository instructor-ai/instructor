from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from instructor.processing.response import handle_response_model
from instructor.providers.bedrock.utils import _prepare_bedrock_converse_kwargs_internal
from instructor.providers.openai.utils import (
    handle_responses_tools,
    handle_responses_tools_with_inbuilt_tools,
)


def test_typed_dict_conversion() -> None:
    class User(TypedDict):  # type: ignore
        name: str
        age: int

    _, user_tool_definition = handle_response_model(User)

    class User(BaseModel):
        name: str
        age: int

    _, pydantic_user_tool_definition = handle_response_model(User)
    assert user_tool_definition == pydantic_user_tool_definition


def test_openai_to_bedrock_conversion() -> None:
    """OpenAI-style input should be fully converted to Bedrock format."""
    call_kwargs = {
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Extract: Jason is 22 years old"},
            {"role": "assistant", "content": "Sure! Jason is 22."},
        ],
    }
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert "model" not in result
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [
        {"text": "Extract: Jason is 22 years old"}
    ]
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == [{"text": "Sure! Jason is 22."}]


def test_bedrock_native_preserved() -> None:
    """Bedrock-native input should be preserved as-is."""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "You are a helpful assistant."}],
        "messages": [
            {"role": "user", "content": [{"text": "Extract: Jason is 22 years old"}]},
            {"role": "assistant", "content": [{"text": "Sure! Jason is 22."}]},
        ],
    }
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == [
        {"text": "Extract: Jason is 22 years old"}
    ]
    assert result["messages"][1]["content"] == [{"text": "Sure! Jason is 22."}]


def test_mixed_openai_and_bedrock() -> None:
    """Mixed input: OpenAI-style is converted, Bedrock-native is preserved."""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "You are a helpful assistant."}],
        "messages": [
            {
                "role": "user",
                "content": "Extract: Jason is 22 years old",
            },  # OpenAI style
            {
                "role": "assistant",
                "content": [{"text": "Sure! Jason is 22."}],
            },  # Bedrock style
        ],
    }
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 2
    # OpenAI-style user message converted
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["messages"][0]["content"] == [
        {"text": "Extract: Jason is 22 years old"}
    ]
    # Bedrock-style assistant message preserved
    assert result["messages"][1]["content"] == [{"text": "Sure! Jason is 22."}]


def test_bedrock_round_trip() -> None:
    """Bedrock input should be unchanged after round-trip through the function."""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "Bedrock system."}],
        "messages": [
            {"role": "user", "content": [{"text": "Bedrock user message."}]},
        ],
    }
    import copy

    original = copy.deepcopy(call_kwargs)
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result == original


def test_empty_and_missing_content() -> None:
    """Empty messages and missing content should be handled gracefully."""
    # Empty messages
    call_kwargs = {"messages": []}
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["messages"] == []
    # Message with no content
    call_kwargs = {"messages": [{"role": "user"}]}
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    assert result["messages"][0]["role"] == "user"
    # Should not add a content key if not present
    assert "content" not in result["messages"][0]


def test_bedrock_invalid_content_format() -> None:
    """Invalid content types should raise ValueError."""
    call_kwargs = {
        "messages": [{"role": "user", "content": 12345}]  # Invalid content type
    }
    try:
        _prepare_bedrock_converse_kwargs_internal(call_kwargs)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unsupported message content type for Bedrock" in str(e)


def test_handle_responses_tools_includes_description() -> None:
    """Tool description must be included in the RESPONSES_TOOLS payload."""

    class User(BaseModel):
        """Extract a user from text."""

        name: str = Field(description="The user's full name")
        age: int

    _, new_kwargs = handle_responses_tools(User, {})
    tool = new_kwargs["tools"][0]
    assert "description" in tool, "tool description missing from RESPONSES_TOOLS payload"
    assert tool["description"] == "Extract a user from text."


def test_handle_responses_tools_fallback_description() -> None:
    """Models without a docstring get a generated description."""

    class Item(BaseModel):
        title: str

    _, new_kwargs = handle_responses_tools(Item, {})
    tool = new_kwargs["tools"][0]
    assert "description" in tool
    assert "Item" in tool["description"]


def test_handle_responses_tools_matches_inbuilt_tools_format() -> None:
    """RESPONSES_TOOLS and RESPONSES_TOOLS_WITH_INBUILT_TOOLS must produce
    the same tool definition for an identical response model."""

    class Order(BaseModel):
        """Process an order."""

        product: str
        quantity: int

    _, kwargs_plain = handle_responses_tools(Order, {})
    _, kwargs_inbuilt = handle_responses_tools_with_inbuilt_tools(Order, {})

    plain_tool = kwargs_plain["tools"][0]
    inbuilt_tool = kwargs_inbuilt["tools"][0]

    assert plain_tool == inbuilt_tool

from typing_extensions import TypedDict
from pydantic import BaseModel
from instructor.process_response import handle_response_model, _prepare_bedrock_converse_kwargs_internal


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


def test_bedrock_system_parameter_conversion() -> None:
    """Test conversion of Bedrock-native system=[{'text': '...'}] to OpenAI format"""
    call_kwargs = {
        "system": [{'text': 'You are a helpful assistant.'}],
        "messages": [{'role': 'user', 'content': 'Hello'}]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # System parameter should be removed and converted to system message in messages
    assert "system" not in result
    assert len(result["messages"]) == 2
    assert result["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert result["messages"][1] == {'role': 'user', 'content': 'Hello'}


def test_bedrock_content_list_format() -> None:
    """Test handling of content=[{'text': '...'}] format in messages"""
    call_kwargs = {
        "messages": [
            {'role': 'user', 'content': [{'text': "Extract: Jason is 22 years old"}]}
        ]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Content should remain in list format for Bedrock
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == [{'text': "Extract: Jason is 22 years old"}]


def test_bedrock_combined_system_and_content_format() -> None:
    """Test the exact issue scenario: system=[{'text': '...'}] + content=[{'text': '...'}]"""
    call_kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{'text': 'You are a helpful assistant.'}],
        "messages": [{'role': 'user', 'content': [{'text': "Extract: Jason is 22 years old"}]}]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should convert system parameter and preserve content format
    assert "system" not in result
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert len(result["messages"]) == 2
    assert result["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][1]["content"] == [{'text': "Extract: Jason is 22 years old"}]


def test_bedrock_backward_compatibility_openai_format() -> None:
    """Test that existing OpenAI format still works"""
    call_kwargs = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Extract: Jason is 22 years old"}
        ]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should process OpenAI format correctly (string content converted to text objects)
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "You are a helpful assistant."  # System content stays as string
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][1]["content"] == [{"text": "Extract: Jason is 22 years old"}]


def test_bedrock_model_id_conversion() -> None:
    """Test that model parameter is converted to modelId"""
    call_kwargs = {
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "messages": []
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    assert "model" not in result
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"


def test_bedrock_invalid_content_format() -> None:
    """Test that invalid content formats still raise NotImplementedError"""
    call_kwargs = {
        "messages": [
            {'role': 'user', 'content': 12345}  # Invalid content type
        ]
    }
    
    try:
        _prepare_bedrock_converse_kwargs_internal(call_kwargs)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "Non-text prompts are not currently supported" in str(e)

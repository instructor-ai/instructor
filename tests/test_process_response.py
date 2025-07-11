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
    """Test conversion of Bedrock-native system=[{'text': '...'}] parameter with OpenAI message content"""
    call_kwargs = {
        "system": [{'text': 'You are a helpful assistant.'}],
        "messages": [{'role': 'user', 'content': 'Hello'}]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should convert to proper Bedrock format
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [{"text": "Hello"}]


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
    """Test OpenAI format input converted to Bedrock format output"""
    # OpenAI-style input
    call_kwargs = {
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Extract: Jason is 22 years old"}
        ]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should convert to Bedrock format
    assert "model" not in result
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [{'text': "Extract: Jason is 22 years old"}]


def test_bedrock_backward_compatibility_openai_format() -> None:
    """Test that OpenAI format gets converted properly to Bedrock format"""
    call_kwargs = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Extract: Jason is 22 years old"}
        ]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should convert to Bedrock format: system extracted, user content as text objects
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [{"text": "Extract: Jason is 22 years old"}]


def test_bedrock_mixed_format_scenarios() -> None:
    """Test mixed format: some args already in Bedrock format, others in OpenAI format"""
    # Mixed format: Bedrock system parameter + OpenAI messages
    call_kwargs = {
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{'text': 'You are a helpful assistant.'}],  # Already Bedrock format
        "messages": [
            {"role": "user", "content": "Extract: Jason is 22 years old"}  # OpenAI format
        ]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should handle mixed format correctly
    assert "model" not in result
    assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [{'text': "Extract: Jason is 22 years old"}]


def test_bedrock_mixed_format_content() -> None:
    """Test mixed content formats: some already as text objects, others as strings"""
    call_kwargs = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},  # OpenAI string format
            {"role": "user", "content": [{'text': "Extract: Jason is 22 years old"}]}  # Already Bedrock format
        ]
    }
    
    result = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    
    # Should handle mixed content formats
    assert result["system"] == [{"text": "You are a helpful assistant."}]
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == [{'text': "Extract: Jason is 22 years old"}]


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

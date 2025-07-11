"""
Test cases for GenAI client with response_model=None.

This test verifies that the GenAI client properly handles the case when
response_model is set to None, ensuring that OpenAI-style messages are 
correctly converted to GenAI-style contents.
"""

import pytest
from instructor.mode import Mode


@pytest.mark.parametrize("mode", [Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS])
def test_genai_response_model_none(genai_client, mode):
    """Test that GenAI client works with response_model=None"""
    
    # This should not raise a "Models.generate_content() got an unexpected keyword argument 'messages'" error
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
    
    # This should work without error and return the raw response
    response = genai_client.chat.completions.create(
        messages=messages,
        response_model=None,
        mode=mode
    )
    
    # We expect to get back a response object, not a parsed model
    assert response is not None
    # The response should not be a Pydantic model since response_model=None
    from pydantic import BaseModel
    assert not isinstance(response, BaseModel)


def test_genai_response_model_none_with_system_message(genai_client):
    """Test that GenAI client works with response_model=None and system message"""
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
    
    # This should work without error and properly extract system message
    response = genai_client.chat.completions.create(
        messages=messages,
        response_model=None,
        mode=Mode.GENAI_TOOLS
    )
    
    assert response is not None
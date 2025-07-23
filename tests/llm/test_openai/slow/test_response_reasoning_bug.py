import pytest
from unittest.mock import Mock
import instructor
from openai import OpenAI
from pydantic import BaseModel


class UserTestModel(BaseModel):
    name: str
    value: int


def test_response_reasoning_item_arguments_bug():
    """
    Test that reproduces the bug where ResponseReasoningItem objects
    don't have 'arguments' attribute, causing AttributeError in reask_responses_tools.
    
    This test reproduces issue #1741 by creating a mock response with both
    ResponseReasoningItem and ResponseFunctionToolCall objects.
    """
    from openai.types.responses import ResponseReasoningItem, ResponseFunctionToolCall
    
    mock_reasoning_item = Mock(spec=ResponseReasoningItem)
    mock_reasoning_item.id = "rs_123"
    mock_reasoning_item.type = "reasoning"
    
    mock_function_call = Mock(spec=ResponseFunctionToolCall)
    mock_function_call.arguments = '{"name": "test", "value": 42}'
    mock_function_call.name = "UserTestModel"
    mock_function_call.type = "function_call"
    
    mock_response = Mock()
    mock_response.output = [mock_reasoning_item, mock_function_call]
    
    validation_error = Exception("Validation failed")
    
    from instructor.providers.openai.utils import reask_responses_tools
    
    kwargs = {
        "messages": [{"role": "user", "content": "test"}],
        "model": "o3",
        "response_model": UserTestModel
    }
    
    with pytest.raises(AttributeError, match="Mock object has no attribute 'arguments'"):
        reask_responses_tools(kwargs, mock_response, validation_error)


def test_response_reasoning_item_with_instructor_client():
    """
    Integration test that reproduces the bug using instructor client with mocked OpenAI response.
    This simulates the actual scenario described in issue #1741.
    """
    from openai.types.responses import ResponseReasoningItem, ResponseFunctionToolCall
    from unittest.mock import patch
    
    mock_client = Mock(spec=OpenAI)
    
    instructor_client = instructor.from_openai(
        mock_client, 
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
    )
    
    mock_reasoning_item = Mock(spec=ResponseReasoningItem)
    mock_reasoning_item.id = "rs_123"
    mock_reasoning_item.type = "reasoning"
    
    mock_function_call = Mock(spec=ResponseFunctionToolCall)
    mock_function_call.arguments = '{"name": "test", "value": 42}'
    mock_function_call.name = "UserTestModel"
    mock_function_call.type = "function_call"
    
    mock_response = Mock()
    mock_response.output = [mock_reasoning_item, mock_function_call]
    
    with patch.object(mock_client.responses, 'create') as mock_create:
        mock_create.return_value = mock_response
        
        try:
            result = instructor_client.responses.create(
                model="o3",
                input="Generate a test model",
                response_model=UserTestModel,
                max_retries=1
            )
        except AttributeError as e:
            assert "'ResponseReasoningItem' object has no attribute 'arguments'" in str(e)
        else:
            pass

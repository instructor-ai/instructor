"""
Test script to verify the Gemini thought parts filtering fix works correctly.
This creates a mock response with thought parts to test the parse_genai_tools method.
"""
import pytest
from unittest.mock import Mock, patch
from instructor.function_calls import OpenAISchema


class UserModel(OpenAISchema):
    name: str
    age: int


@patch('instructor.function_calls.isinstance')
def test_genai_tools_with_thought_parts(mock_isinstance):
    """Test that parse_genai_tools correctly filters out thought parts."""
    mock_isinstance.return_value = True
    
    mock_completion = Mock()
    mock_completion.candidates = [Mock()]
    mock_completion.candidates[0].content = Mock()
    
    thought_part = Mock()
    thought_part.thought = True
    
    function_call_part = Mock()
    function_call_part.thought = False
    function_call_part.function_call = Mock()
    function_call_part.function_call.name = "UserModel"
    function_call_part.function_call.args = {"name": "John", "age": 30}
    
    mock_completion.candidates[0].content.parts = [thought_part, function_call_part]
    
    result = UserModel.parse_genai_tools(mock_completion)
    assert result.name == "John"
    assert result.age == 30


@patch('instructor.function_calls.isinstance')
def test_genai_tools_without_thought_parts(mock_isinstance):
    """Test that parse_genai_tools works normally without thought parts."""
    mock_isinstance.return_value = True
    
    mock_completion = Mock()
    mock_completion.candidates = [Mock()]
    mock_completion.candidates[0].content = Mock()
    
    function_call_part = Mock(spec=[])  # Empty spec to avoid implicit attributes
    function_call_part.function_call = Mock()
    function_call_part.function_call.name = "UserModel"
    function_call_part.function_call.args = {"name": "Jane", "age": 25}
    
    mock_completion.candidates[0].content.parts = [function_call_part]
    
    result = UserModel.parse_genai_tools(mock_completion)
    assert result.name == "Jane"
    assert result.age == 25


@patch('instructor.function_calls.isinstance')
def test_genai_tools_with_multiple_thought_parts(mock_isinstance):
    """Test that parse_genai_tools filters out multiple thought parts."""
    mock_isinstance.return_value = True
    
    mock_completion = Mock()
    mock_completion.candidates = [Mock()]
    mock_completion.candidates[0].content = Mock()
    
    thought_part1 = Mock()
    thought_part1.thought = True
    
    thought_part2 = Mock()
    thought_part2.thought = True
    
    function_call_part = Mock()
    function_call_part.thought = False
    function_call_part.function_call = Mock()
    function_call_part.function_call.name = "UserModel"
    function_call_part.function_call.args = {"name": "Alice", "age": 35}
    
    mock_completion.candidates[0].content.parts = [thought_part1, thought_part2, function_call_part]
    
    result = UserModel.parse_genai_tools(mock_completion)
    assert result.name == "Alice"
    assert result.age == 35


@patch('instructor.function_calls.isinstance')
def test_genai_tools_with_no_thought_attribute(mock_isinstance):
    """Test that parse_genai_tools works with parts that don't have thought attribute."""
    mock_isinstance.return_value = True
    
    mock_completion = Mock()
    mock_completion.candidates = [Mock()]
    mock_completion.candidates[0].content = Mock()
    
    function_call_part = Mock()
    if hasattr(function_call_part, 'thought'):
        delattr(function_call_part, 'thought')
    function_call_part.function_call = Mock()
    function_call_part.function_call.name = "UserModel"
    function_call_part.function_call.args = {"name": "Bob", "age": 40}
    
    mock_completion.candidates[0].content.parts = [function_call_part]
    
    result = UserModel.parse_genai_tools(mock_completion)
    assert result.name == "Bob"
    assert result.age == 40


@patch('instructor.function_calls.isinstance')
def test_genai_tools_multiple_function_calls_should_fail(mock_isinstance):
    """Test that parse_genai_tools still fails appropriately with multiple function calls."""
    mock_isinstance.return_value = True
    
    mock_completion = Mock()
    mock_completion.candidates = [Mock()]
    mock_completion.candidates[0].content = Mock()
    
    function_call_part1 = Mock()
    function_call_part1.thought = False
    function_call_part1.function_call = Mock()
    function_call_part1.function_call.name = "UserModel"
    function_call_part1.function_call.args = {"name": "Charlie", "age": 45}
    
    function_call_part2 = Mock()
    function_call_part2.thought = False
    function_call_part2.function_call = Mock()
    function_call_part2.function_call.name = "UserModel"
    function_call_part2.function_call.args = {"name": "David", "age": 50}
    
    mock_completion.candidates[0].content.parts = [function_call_part1, function_call_part2]
    
    with pytest.raises(AssertionError, match="Instructor does not support multiple function calls"):
        UserModel.parse_genai_tools(mock_completion)


@patch('instructor.function_calls.isinstance')
def test_genai_tools_thought_parts_mixed_with_multiple_function_calls(mock_isinstance):
    """Test filtering thought parts but still failing on multiple function calls."""
    mock_isinstance.return_value = True
    
    mock_completion = Mock()
    mock_completion.candidates = [Mock()]
    mock_completion.candidates[0].content = Mock()
    
    thought_part = Mock()
    thought_part.thought = True
    
    function_call_part1 = Mock()
    function_call_part1.thought = False
    function_call_part1.function_call = Mock()
    function_call_part1.function_call.name = "UserModel"
    function_call_part1.function_call.args = {"name": "Eve", "age": 55}
    
    function_call_part2 = Mock()
    function_call_part2.thought = False
    function_call_part2.function_call = Mock()
    function_call_part2.function_call.name = "UserModel"
    function_call_part2.function_call.args = {"name": "Frank", "age": 60}
    
    mock_completion.candidates[0].content.parts = [thought_part, function_call_part1, function_call_part2]
    
    with pytest.raises(AssertionError, match="Instructor does not support multiple function calls"):
        UserModel.parse_genai_tools(mock_completion)

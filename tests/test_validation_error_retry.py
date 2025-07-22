"""Test that validation errors properly trigger retry mechanism."""

import pytest
from unittest.mock import Mock, patch
from pydantic import BaseModel, ValidationError as PydanticValidationError
from instructor.core.exceptions import ValidationError as InstructorValidationError, InstructorRetryException
from instructor.core.retry import retry_sync, retry_async
from instructor.mode import Mode


class TestModel(BaseModel):
    name: str
    age: int


def test_retry_sync_handles_pydantic_validation_error():
    """Test that retry_sync catches pydantic ValidationError and calls handle_reask_kwargs."""
    def mock_func(*args, **kwargs):
        raise PydanticValidationError.from_exception_data("TestModel", [])
    
    with patch('instructor.core.retry.handle_reask_kwargs') as mock_handle_reask:
        mock_handle_reask.return_value = {"messages": []}
        
        with pytest.raises(InstructorRetryException):
            retry_sync(
                func=mock_func,
                response_model=TestModel,
                args=(),
                kwargs={"messages": []},
                max_retries=1,
                mode=Mode.JSON
            )
        
        assert mock_handle_reask.called


def test_retry_sync_handles_instructor_validation_error():
    """Test that retry_sync catches instructor ValidationError and calls handle_reask_kwargs."""
    def mock_func(*args, **kwargs):
        raise InstructorValidationError("Test validation error")
    
    with patch('instructor.core.retry.handle_reask_kwargs') as mock_handle_reask:
        mock_handle_reask.return_value = {"messages": []}
        
        with pytest.raises(InstructorRetryException):
            retry_sync(
                func=mock_func,
                response_model=TestModel,
                args=(),
                kwargs={"messages": []},
                max_retries=1,
                mode=Mode.JSON
            )
        
        assert mock_handle_reask.called


@pytest.mark.asyncio
async def test_retry_async_handles_pydantic_validation_error():
    """Test that retry_async catches pydantic ValidationError and calls handle_reask_kwargs."""
    async def mock_func(*args, **kwargs):
        raise PydanticValidationError.from_exception_data("TestModel", [])
    
    with patch('instructor.core.retry.handle_reask_kwargs') as mock_handle_reask:
        mock_handle_reask.return_value = {"messages": []}
        
        with pytest.raises(InstructorRetryException):
            await retry_async(
                func=mock_func,
                response_model=TestModel,
                args=(),
                kwargs={"messages": []},
                max_retries=1,
                mode=Mode.JSON
            )
        
        assert mock_handle_reask.called


@pytest.mark.asyncio
async def test_retry_async_handles_instructor_validation_error():
    """Test that retry_async catches instructor ValidationError and calls handle_reask_kwargs."""
    async def mock_func(*args, **kwargs):
        raise InstructorValidationError("Test validation error")
    
    with patch('instructor.core.retry.handle_reask_kwargs') as mock_handle_reask:
        mock_handle_reask.return_value = {"messages": []}
        
        with pytest.raises(InstructorRetryException):
            await retry_async(
                func=mock_func,
                response_model=TestModel,
                args=(),
                kwargs={"messages": []},
                max_retries=1,
                mode=Mode.JSON
            )
        
        assert mock_handle_reask.called


def test_both_validation_errors_inherit_from_base_exception():
    """Test that both ValidationError types can be caught by their base classes."""
    pydantic_error = PydanticValidationError.from_exception_data("TestModel", [])
    instructor_error = InstructorValidationError("Test error")
    
    assert isinstance(pydantic_error, Exception)
    assert isinstance(instructor_error, Exception)
    assert isinstance(instructor_error, InstructorValidationError)

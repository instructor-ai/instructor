"""Test that all instructor exceptions can be imported and caught properly."""

import pytest
from instructor.exceptions import (
    InstructorError,
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
    ProviderError,
    ConfigurationError,
    ModeError,
    ClientError,
)


def test_all_exceptions_can_be_imported():
    """Test that all exceptions can be imported from instructor base package"""
    # This test passes if the imports above succeed
    assert InstructorError is not None
    assert IncompleteOutputException is not None
    assert InstructorRetryException is not None
    assert ValidationError is not None
    assert ProviderError is not None
    assert ConfigurationError is not None
    assert ModeError is not None
    assert ClientError is not None


def test_exception_hierarchy():
    """Test that all exceptions inherit from InstructorError."""
    assert issubclass(IncompleteOutputException, InstructorError)
    assert issubclass(InstructorRetryException, InstructorError)
    assert issubclass(ValidationError, InstructorError)
    assert issubclass(ProviderError, InstructorError)
    assert issubclass(ConfigurationError, InstructorError)
    assert issubclass(ModeError, InstructorError)
    assert issubclass(ClientError, InstructorError)


def test_base_instructor_error_can_be_caught():
    """Test that InstructorError can catch all instructor exceptions."""
    with pytest.raises(InstructorError):
        raise IncompleteOutputException()

    with pytest.raises(InstructorError):
        raise InstructorRetryException(n_attempts=3, total_usage=100)

    with pytest.raises(InstructorError):
        raise ValidationError("Validation failed")

    with pytest.raises(InstructorError):
        raise ProviderError("openai", "API error")

    with pytest.raises(InstructorError):
        raise ConfigurationError("Invalid config")

    with pytest.raises(InstructorError):
        raise ModeError("tools", "openai", ["json"])

    with pytest.raises(InstructorError):
        raise ClientError("Client initialization failed")


def test_incomplete_output_exception():
    """Test IncompleteOutputException attributes and catching."""
    last_completion = {"content": "partial response"}

    with pytest.raises(IncompleteOutputException) as exc_info:
        raise IncompleteOutputException(last_completion=last_completion)

    assert exc_info.value.last_completion == last_completion
    assert "incomplete due to a max_tokens length limit" in str(exc_info.value)


def test_instructor_retry_exception():
    """Test InstructorRetryException attributes and catching."""
    last_completion = {"content": "failed response"}
    messages = [{"role": "user", "content": "test"}]
    n_attempts = 3
    total_usage = 150
    create_kwargs = {"model": "gpt-3.5-turbo"}

    with pytest.raises(InstructorRetryException) as exc_info:
        raise InstructorRetryException(
            last_completion=last_completion,
            messages=messages,
            n_attempts=n_attempts,
            total_usage=total_usage,
            create_kwargs=create_kwargs,
        )

    exception = exc_info.value
    assert exception.last_completion == last_completion
    assert exception.messages == messages
    assert exception.n_attempts == n_attempts
    assert exception.total_usage == total_usage
    assert exception.create_kwargs == create_kwargs


def test_validation_error():
    """Test ValidationError can be caught."""
    error_message = "Field validation failed"

    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError(error_message)

    assert str(exc_info.value) == error_message


def test_provider_error():
    """Test ProviderError attributes and catching."""
    provider = "anthropic"
    message = "Rate limit exceeded"

    with pytest.raises(ProviderError) as exc_info:
        raise ProviderError(provider, message)

    exception = exc_info.value
    assert exception.provider == provider
    assert f"{provider}: {message}" in str(exception)


def test_configuration_error():
    """Test ConfigurationError can be caught."""
    error_message = "Missing required configuration"

    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError(error_message)

    assert str(exc_info.value) == error_message


def test_mode_error():
    """Test ModeError attributes and catching."""
    mode = "invalid_mode"
    provider = "openai"
    valid_modes = ["json", "tools", "functions"]

    with pytest.raises(ModeError) as exc_info:
        raise ModeError(mode, provider, valid_modes)

    exception = exc_info.value
    assert exception.mode == mode
    assert exception.provider == provider
    assert exception.valid_modes == valid_modes
    assert f"Invalid mode '{mode}' for provider '{provider}'" in str(exception)
    assert "json, tools, functions" in str(exception)


def test_client_error():
    """Test ClientError can be caught."""
    error_message = "Client not properly initialized"

    with pytest.raises(ClientError) as exc_info:
        raise ClientError(error_message)

    assert str(exc_info.value) == error_message


def test_specific_exception_catching():
    """Test that specific exceptions can be caught individually."""
    # Test that we can catch specific exceptions without catching others

    with pytest.raises(IncompleteOutputException):
        try:
            raise IncompleteOutputException()
        except InstructorRetryException:
            pytest.fail("Should not catch InstructorRetryException")
        except IncompleteOutputException:
            raise  # Re-raise to be caught by pytest.raises

    with pytest.raises(ProviderError):
        try:
            raise ProviderError("test", "error")
        except ConfigurationError:
            pytest.fail("Should not catch ConfigurationError")
        except ProviderError:
            raise  # Re-raise to be caught by pytest.raises


def test_multiple_exception_handling():
    """Test handling multiple exception types in a single try-except block."""

    def raise_exception(exc_type: str):
        if exc_type == "incomplete":
            raise IncompleteOutputException()
        elif exc_type == "retry":
            raise InstructorRetryException(n_attempts=3, total_usage=100)
        elif exc_type == "validation":
            raise ValidationError("validation failed")
        else:
            raise ValueError("unknown exception type")

    # Test catching multiple specific exceptions
    for exc_type in ["incomplete", "retry", "validation"]:
        with pytest.raises(
            (IncompleteOutputException, InstructorRetryException, ValidationError)
        ):
            raise_exception(exc_type)

    # Test that base exception catches all instructor exceptions
    for exc_type in ["incomplete", "retry", "validation"]:
        with pytest.raises(InstructorError):
            raise_exception(exc_type)

    # Test that non-instructor exceptions are not caught
    with pytest.raises(ValueError):
        raise_exception("unknown")


def test_exception_import_from_instructor():
    """Test that exceptions can be imported from the main instructor module."""
    # Test importing from instructor.exceptions (already done in module imports)
    from instructor.exceptions import InstructorError as ImportedError

    assert ImportedError is InstructorError

    # Test that exceptions are accessible and can be used in real scenarios
    try:
        raise ImportedError("test error")
    except InstructorError as e:
        assert str(e) == "test error"

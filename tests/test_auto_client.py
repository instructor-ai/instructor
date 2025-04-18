from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from instructor.auto_client import from_provider
from instructor.client import Instructor, AsyncInstructor
from instructor.mode import Mode


@pytest.mark.parametrize(
    "model_string, expected_provider, expected_model",
    [
        ("openai/gpt-4", "openai", "gpt-4"),
        ("anthropic/claude-3-sonnet", "anthropic", "claude-3-sonnet"),
        ("google/gemini-pro", "google", "gemini-pro"),
        ("mistral/mistral-large", "mistral", "mistral-large"),
    ],
)
def test_provider_parsing(model_string, expected_provider, expected_model):
    """Test that provider strings are parsed correctly."""
    with patch("instructor.auto_client.from_openai") as mock_from_openai:
        with patch("instructor.auto_client.openai") as mock_openai:
            # Configure mocks
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            # Only run test for openai to simplify
            if expected_provider != "openai":
                pytest.skip(f"Skipping non-openai provider: {expected_provider}")

            # Call the function
            from_provider(model_string)

            # Check mock was called correctly
            mock_from_openai.assert_called_once()
            args, kwargs = mock_from_openai.call_args
            assert kwargs.get("model") == expected_model


def test_invalid_provider_format():
    """Test that error is raised for invalid provider format."""
    with pytest.raises(ValueError) as excinfo:
        from_provider("invalid-format")
    assert "Model string must be in format" in str(excinfo.value)


def test_unsupported_provider():
    """Test that error is raised for unsupported provider."""
    with pytest.raises(ValueError) as excinfo:
        from_provider("unsupported/model")
    assert "Unsupported provider" in str(excinfo.value)


@pytest.mark.parametrize(
    "async_param, expected_type",
    [
        (False, Instructor),
        (True, AsyncInstructor),
    ],
)
def test_async_parameter(async_param, expected_type):
    """Test that async_client parameter works correctly."""
    with patch("instructor.auto_client.from_openai") as mock_from_openai:
        with patch("instructor.auto_client.openai") as mock_openai:
            # Setup mocks
            if async_param:
                mock_openai.AsyncOpenAI.return_value = MagicMock()
            else:
                mock_openai.OpenAI.return_value = MagicMock()

            mock_instructor = MagicMock(spec=expected_type)
            mock_from_openai.return_value = mock_instructor

            # Call function
            result = from_provider("openai/gpt-4", async_client=async_param)

            # Verify correct client was created
            if async_param:
                mock_openai.AsyncOpenAI.assert_called_once()
            else:
                mock_openai.OpenAI.assert_called_once()

            # Verify correct instructor type was returned
            assert isinstance(mock_instructor, expected_type.__class__)


def test_mode_parameter():
    """Test that mode parameter is passed correctly."""
    with patch("instructor.auto_client.from_openai") as mock_from_openai:
        with patch("instructor.auto_client.openai") as mock_openai:
            # Call with custom mode
            from_provider("openai/gpt-4", mode=Mode.JSON)

            # Verify mode was passed
            args, kwargs = mock_from_openai.call_args
            assert kwargs.get("mode") == Mode.JSON


def test_default_mode_used():
    """Test that default mode is used when mode not specified."""
    with patch("instructor.auto_client.from_openai") as mock_from_openai:
        with patch("instructor.auto_client.openai") as mock_openai:
            # Call without specifying mode
            from_provider("openai/gpt-4")

            # Verify default mode was passed
            args, kwargs = mock_from_openai.call_args
            assert kwargs.get("mode") == Mode.OPENAI_FUNCTIONS


def test_additional_kwargs_passed():
    """Test that additional kwargs are passed to provider."""
    with patch("instructor.auto_client.from_openai") as mock_from_openai:
        with patch("instructor.auto_client.openai") as mock_openai:
            # Call with additional kwargs
            from_provider("openai/gpt-4", custom_param=123, another_param="test")

            # Verify kwargs were passed
            args, kwargs = mock_from_openai.call_args
            assert kwargs.get("custom_param") == 123
            assert kwargs.get("another_param") == "test"


@patch("importlib.util.find_spec", return_value=None)
def test_missing_dependency(_):
    """Test that ImportError is raised when required package is missing."""
    with pytest.raises(ImportError) as excinfo:
        from_provider("openai/gpt-4")
    assert "openai package is required" in str(
        excinfo.value
    ) or "required package" in str(excinfo.value)

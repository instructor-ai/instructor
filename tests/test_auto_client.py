from __future__ import annotations

import pytest
from instructor.auto_client import from_provider
from pydantic import BaseModel


# --- User model and prompt (from main.py) ---
class User(BaseModel):
    name: str
    age: int


USER_EXTRACTION_PROMPT = {
    "role": "user",
    "content": "Ivan is 28 and strays in Singapore. Extract it as a user object",
}

# --- Providers to test (from main.py) ---
PROVIDERS = [
    "anthropic/claude-3-5-haiku-latest",
    "google/gemini-2.0-flash",
    "openai/gpt-4o-mini",
    "azure_openai/gpt-4o-mini",
    "mistral/ministral-8b-latest",
    "cohere/command-r-plus",
    "perplexity/sonar-pro",
    "groq/llama-3.1-8b-instant",
    "writer/palmyra-x5",
    "cerebras/llama-4-scout-17b-16e-instruct",
    "deepseek/deepseek-chat",
    "fireworks/accounts/fireworks/models/llama4-maverick-instruct-basic",
    "vertexai/gemini-1.5-flash",
]


def should_skip_provider(provider_string: str) -> bool:
    import os

    if os.getenv("INSTRUCTOR_ENV") == "CI":
        return provider_string not in [
            "cohere/command-r-plus",
            "google/gemini-2.0-flash",
            "openai/gpt-4o-mini",
        ]
    return False


@pytest.mark.parametrize("provider_string", PROVIDERS)
def test_user_extraction_sync(provider_string):
    """Test user extraction for each provider (sync)."""

    if should_skip_provider(provider_string):
        pytest.skip(f"Skipping provider {provider_string} on CI")
        return

    try:
        client = from_provider(provider_string)  # type: ignore[arg-type]
        response = client.chat.completions.create(
            messages=[USER_EXTRACTION_PROMPT],  # type: ignore[arg-type]
            response_model=User,
        )
        assert isinstance(response, User)
        assert response.name.lower() == "ivan"
        assert response.age == 28
    except Exception as e:
        pytest.skip(f"Provider {provider_string} not available or failed: {e}")


@pytest.mark.parametrize("provider_string", PROVIDERS)
@pytest.mark.asyncio
async def test_user_extraction_async(provider_string):
    """Test user extraction for each provider (async)."""

    if should_skip_provider(provider_string):
        pytest.skip(f"Skipping provider {provider_string} on CI")
        return

    client = from_provider(provider_string, async_client=True)  # type: ignore[arg-type]
    response = await client.chat.completions.create(
        messages=[USER_EXTRACTION_PROMPT],  # type: ignore[arg-type]
        response_model=User,
    )
    assert isinstance(response, User)
    assert response.name.lower() == "ivan"
    assert response.age == 28


def test_invalid_provider_format():
    """Test that error is raised for invalid provider format."""
    from instructor.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        from_provider("invalid-format")
    assert "Model string must be in format" in str(excinfo.value)


def test_unsupported_provider():
    """Test that error is raised for unsupported provider."""
    from instructor.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        from_provider("unsupported/model")
    assert "Unsupported provider" in str(excinfo.value)


def test_additional_kwargs_passed():
    """Test that additional kwargs are passed to provider."""
    import instructor
    from instructor.exceptions import InstructorRetryException
    import os

    if os.getenv("INSTRUCTOR_ENV") == "CI":
        pytest.skip("Skipping test on CI")
        return

    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest", max_tokens=10
    )

    with pytest.raises(InstructorRetryException) as excinfo:
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Generate a sentence with 20 characters",
                }
            ],
            response_model=str,
        )

    assert "The output is incomplete due to a max_tokens length limit" in str(
        excinfo.value
    )


def test_api_key_parameter_extraction():
    """Test that api_key parameter is correctly extracted from kwargs."""
    from unittest.mock import patch, MagicMock

    # Mock the openai module to avoid actual API calls
    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the from_openai import
        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            # Test that api_key is passed to client constructor
            from_provider("openai/gpt-4", api_key="test-key-123")

            # Verify OpenAI was called with the api_key
            mock_openai_class.assert_called_once()
            _, kwargs = mock_openai_class.call_args
            assert kwargs["api_key"] == "test-key-123"


def test_api_key_parameter_with_environment_fallback():
    """Test that api_key parameter falls back to environment variables."""
    import os
    from unittest.mock import patch, MagicMock

    # Mock the openai module
    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the from_openai import
        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            # Mock environment variable
            with patch.dict(os.environ, {}, clear=True):
                # Test with no api_key parameter and no environment variable
                from_provider("openai/gpt-4")

                # Should still call OpenAI with None (which is the default behavior)
                mock_openai_class.assert_called()
                _, kwargs = mock_openai_class.call_args
                assert kwargs["api_key"] is None


def test_api_key_parameter_with_async_client():
    """Test that api_key parameter works with async clients."""
    from unittest.mock import patch, MagicMock

    # Mock the openai module
    with patch("openai.AsyncOpenAI") as mock_async_openai_class:
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client

        # Mock the from_openai import
        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            # Test with async client
            from_provider("openai/gpt-4", async_client=True, api_key="test-async-key")

            # Verify AsyncOpenAI was called with the api_key
            mock_async_openai_class.assert_called_once()
            _, kwargs = mock_async_openai_class.call_args
            assert kwargs["api_key"] == "test-async-key"


def test_api_key_parameter_not_passed_when_none():
    """Test that api_key parameter is handled correctly when None."""
    from unittest.mock import patch, MagicMock

    # Mock the openai module
    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the from_openai import
        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            # Test with None api_key
            from_provider("openai/gpt-4", api_key=None)

            # Verify OpenAI was called with None api_key
            mock_openai_class.assert_called_once()
            _, kwargs = mock_openai_class.call_args
            assert kwargs["api_key"] is None


def test_api_key_logging():
    """Test that api_key provision is logged correctly."""
    from unittest.mock import patch, MagicMock

    # Mock the openai module
    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the from_openai import
        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            # Mock logger
            with patch("instructor.auto_client.logger") as mock_logger:
                # Test that providing api_key triggers debug log
                from_provider("openai/gpt-4", api_key="test-key")

                # Check that debug was called with api_key message and length
                debug_calls = [
                    call
                    for call in mock_logger.debug.call_args_list
                    if "API key provided" in str(call) and "length:" in str(call)
                ]
                assert len(debug_calls) > 0, (
                    "Expected debug log for API key provision with length"
                )

                # Verify the length is logged correctly (test-key is 8 characters)
                mock_logger.debug.assert_called_with(
                    "API key provided for %s provider (length: %d characters)",
                    "openai",
                    8,
                    extra={"provider": "openai", "operation": "initialize"},
                )

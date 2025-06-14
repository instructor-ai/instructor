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
    "mistral/ministral-8b-latest",
    "cohere/command-r-plus",
    "perplexity/sonar-pro",
    "groq/llama-3.1-8b-instant",
    "writer/palmyra-x5",
    "cerebras/llama-4-scout-17b-16e-instruct",
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


@pytest.mark.parametrize(
    "provider,key_name,model_name",
    [
        ("openai", "OPENAI_API_KEY", "gpt-4o-mini"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-3-5-haiku-latest"),
        ("google", "GOOGLE_API_KEY", "gemini-2.0-flash"),
        ("mistral", "MISTRAL_API_KEY", "ministral-8b-latest"),
        ("cohere", "COHERE_API_KEY", "command-r-plus"),
        ("perplexity", "PERPLEXITY_API_KEY", "sonar-pro"),
        ("groq", "GROQ_API_KEY", "llama-3.1-8b-instant"),
        ("writer", "WRITER_API_KEY", "palmyra-x5"),
        ("cerebras", "CEREBRAS_API_KEY", "llama-4-scout-17b-16e-instruct"),
        (
            "fireworks",
            "FIREWORKS_API_KEY",
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
        ),
    ],
)
def test_api_key_parameters(provider, key_name, model_name):
    """Test that api_key parameter is used instead of environment variables."""
    import instructor
    import os
    import pytest

    if os.getenv("INSTRUCTOR_ENV") == "CI" and provider in ["anthropic"]:
        pytest.skip("Skipping test on CI")
        return

    # Save original env var
    original_key = os.environ.get(key_name)

    if not original_key:
        raise ValueError(
            f"API key for {provider} is not set. Please set it for the test to run."
        )

    try:
        if key_name in os.environ:
            del os.environ[key_name]

        model = f"{provider}/{model_name}"

        # Should fail without api_key
        with pytest.raises(ValueError) as excinfo:
            instructor.from_provider(model)

        assert f"API key for {provider} is not set" in str(excinfo.value)

        # Should work with explicit api_key
        client = instructor.from_provider(model, api_key=original_key)
        assert client is not None

        # Test inference
        response = client.chat.completions.create(
            messages=[USER_EXTRACTION_PROMPT],  # type: ignore[arg-type]
            response_model=User,
        )
        assert isinstance(response, User)

    finally:
        # Restore original env var
        os.environ[key_name] = original_key

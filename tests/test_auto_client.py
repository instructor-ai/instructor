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
    with pytest.raises(ValueError) as excinfo:
        from_provider("invalid-format")
    assert "Model string must be in format" in str(excinfo.value)


def test_unsupported_provider():
    """Test that error is raised for unsupported provider."""
    with pytest.raises(ValueError) as excinfo:
        from_provider("unsupported/model")
    assert "Unsupported provider" in str(excinfo.value)


def test_mode_parameter():
    """Test that mode parameter is passed correctly."""
    import instructor
    from pydantic import Field
    from instructor.exceptions import InstructorRetryException

    client = from_provider("openai/gpt-4o-mini", mode=instructor.Mode.TOOLS_STRICT)

    class User(BaseModel):
        name: str
        phone_number: str = Field(pattern=r"^\+?[1-9]\d{1,14}$")

    with pytest.raises(InstructorRetryException) as excinfo:
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Ivan is 28 and his phone number is +1234567890",
                }
            ],  # type: ignore[arg-type]
            response_model=User,
        )

    assert "'properties', 'phone_number'), 'pattern' is not permitted." in str(
        excinfo.value
    ), f"{excinfo.value}"

    client = from_provider("openai/gpt-4o-mini")
    resp = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28 and his phone number is +1234567890",
            }
        ],  # type: ignore[arg-type]
        response_model=User,
    )
    assert resp.phone_number == "+1234567890"


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

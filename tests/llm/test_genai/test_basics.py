import pytest
import os
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
async def test_genai_async_from_provider(mode):
    """Test Google GenAI async client using from_provider with different modes"""
    client = instructor.from_provider(
        "google/gemini-2.5-flash", mode=mode, async_client=True
    )

    user = await client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
def test_genai_sync_from_provider(mode):
    """Test Google GenAI sync client using from_provider with different modes"""
    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)

    user = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25

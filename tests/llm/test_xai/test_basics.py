import pytest
import os
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("mode", [instructor.Mode.XAI_JSON, instructor.Mode.XAI_TOOLS])
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_async_from_provider(mode):
    """Test xAI async client using from_provider with different modes"""
    client = instructor.from_provider("xai/grok-3-mini", mode=mode, async_client=True)

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


@pytest.mark.parametrize("mode", [instructor.Mode.XAI_JSON, instructor.Mode.XAI_TOOLS])
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_sync_from_provider(mode):
    """Test xAI sync client using from_provider with different modes"""
    client = instructor.from_provider("xai/grok-3-mini", mode=mode)

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

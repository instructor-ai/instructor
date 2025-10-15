from pydantic import BaseModel, Field, field_validator
from instructor import from_cohere
from instructor.core.exceptions import InstructorRetryException
import pytest


class User(BaseModel):
    name: str = Field(..., min_length=5)
    age: int = Field(..., ge=18)

    @field_validator("name")
    def name_must_be_bob(cls, v: str) -> str:  # noqa: ARG002
        raise ValueError("Name must be Bob")


def test_user_creation_retry(client_v1):
    try:
        client = from_cohere(client_v1)
        res = client.chat.completions.create(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=User,
        )

    except Exception as e:
        assert isinstance(e, InstructorRetryException)


@pytest.mark.asyncio()
async def test_user_async_creation_retry(aclient_v1):
    client = from_cohere(aclient_v1)
    try:
        res = await client.chat.completions.create(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=User,
        )
    except Exception as e:
        assert isinstance(e, InstructorRetryException)

from pydantic import BaseModel, Field, field_validator
from instructor import from_cohere
from instructor.exceptions import InstructorRetryException
import pytest


class User(BaseModel):
    name: str = Field(..., min_length=5)
    age: int = Field(..., ge=18)

    @field_validator("name")
    def name_must_be_bob(cls, v: str) -> str:  # noqa: ARG002
        raise ValueError("Name must be Bob")


def test_user_creation_retry(client):
    try:
        client = from_cohere(client)
        res = client.chat.completions.create(
            model="command-r",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=User,
        )

    except Exception as e:
        assert isinstance(e, InstructorRetryException)


@pytest.mark.asyncio()
async def test_user_async_creation_retry(aclient):
    client = from_cohere(aclient)
    try:
        res = await client.chat.completions.create(
            model="command-r",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=User,
        )
    except Exception as e:
        assert isinstance(e, InstructorRetryException)


def test_invalid_model(client):
    try:
        client = from_cohere(client)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=User,
        )
    except Exception as e:
        assert isinstance(e, InstructorRetryException)
        assert e.__cause__.__cause__.args[0] == {
            "message": "model 'gpt-4o-mini' not found, make sure the correct model ID was used and that you have access to the model."
        }


@pytest.mark.asyncio()
async def test_invalid_model_async(aclient):
    try:
        client = from_cohere(aclient)
        res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=User,
        )
    except Exception as e:
        assert isinstance(e, InstructorRetryException)
        assert e.__cause__.__cause__.args[0] == {
            "message": "model 'gpt-4o-mini' not found, make sure the correct model ID was used and that you have access to the model."
        }

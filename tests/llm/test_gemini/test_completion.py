import pytest
from google.generativeai import GenerativeModel
from google.generativeai.types import (
    GenerateContentResponse,
    AsyncGenerateContentResponse,
)
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = GenerativeModel(model_name="gemini-1.5-flash")
    client = instructor.from_gemini(client)

    users, completion = client.messages.create_with_completion(  # type: ignore
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, GenerateContentResponse)


def test_create_with_completion_bool():
    client = GenerativeModel(model_name="gemini-1.5-flash-latest")
    client = instructor.from_gemini(client)

    response, completion = client.messages.create_with_completion(
        messages=[{"role": "user", "content": "Is paris the capital of France?"}],
        response_model=bool,  # type: ignore
    )

    # Verify we got a valid response
    assert isinstance(response, bool)
    assert response is True

    assert isinstance(completion, GenerateContentResponse)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = GenerativeModel(model_name="gemini-1.5-flash-latest")
    client = instructor.from_gemini(client, use_async=True)

    users, completion = await client.messages.create_with_completion(
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, AsyncGenerateContentResponse)

import pytest
from mistralai import Mistral
from mistralai.models.chatcompletionresponse import ChatCompletionResponse
from pydantic import BaseModel
import instructor
import os


class User(BaseModel):
    name: str
    age: int


def test_create_with_completion():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    client = instructor.from_mistral(client, mode=instructor.Mode.MISTRAL_TOOLS)

    users, completion = client.chat.completions.create_with_completion(  # type: ignore
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletionResponse)


def test_create_with_completion_bool():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    client = instructor.from_mistral(client)

    response, completion = client.chat.completions.create_with_completion(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Is paris the capital of France?"}],
        response_model=bool,  # type: ignore
    )

    # Verify we got a valid response
    assert isinstance(response, bool)
    assert response is True

    assert isinstance(completion, ChatCompletionResponse)


@pytest.mark.asyncio
async def test_create_with_completion_async():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    client = instructor.from_mistral(client, use_async=True)

    users, completion = await client.chat.completions.create_with_completion(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Ivan is 28, Jack is 29 and John is 30"}],
        response_model=list[User],
    )

    # Verify we got a valid response
    assert isinstance(users, list)
    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)

    assert isinstance(completion, ChatCompletionResponse)
